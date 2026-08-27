"""Inference-only RWM trajectory selection and NAVSIM PDM evaluation."""

import argparse
import logging
import os
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
from pairdrive.data import FeatureCacheLoader, MetricCacheLoader, load_tokens_csv
from pairdrive.models.agent import RWMInferenceAgent


LOGGER = logging.getLogger("pairdrive.evaluation")
RESULT_COLUMNS = [
    "token",
    "base_model",
    "valid",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "comfort",
    "two_frame_extended_comfort",
    "EPDMS",
    "PDMS",
]
MEAN_METRIC_COLUMNS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "comfort",
    "two_frame_extended_comfort",
    "EPDMS",
    "PDMS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="RWM Lightning checkpoint")
    parser.add_argument("--metric-cache", type=Path, required=True, help="NAVSIM metric-cache root")
    parser.add_argument("--feature-cache", type=Path, required=True, help="Cached RWM inference inputs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the result CSV")
    parser.add_argument("--tokens-csv", type=Path, help="Optional CSV subset containing a token column")
    parser.add_argument("--token-column", default="token")
    parser.add_argument(
        "--expected-base-model",
        choices=("diffusiondrive",),
        help="Reject a cache that explicitly declares a different base model",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="RWM inference device (defaults to CPU; CUDA remains available when explicitly requested)",
    )
    parser.add_argument(
        "--worker",
        choices=("sequential", "thread", "process", "ray"),
        default="process",
        help="NAVSIM worker backend used for CPU-heavy PDM scoring",
    )
    parser.add_argument("--workers", type=int, default=8, help="PDM workers per node")
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=8,
        help="Number of cached scenes ranked by RWM in each inference batch",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        help="Number of independent token shards (defaults to torchrun WORLD_SIZE)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        help="Current shard index (defaults to torchrun RANK)",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        help="PyTorch CPU threads used by this rank",
    )
    parser.add_argument(
        "--ray-distributed",
        action="store_true",
        help="Connect the Ray worker to the cluster described by NAVSIM's Ray environment variables",
    )
    parser.add_argument("--max-scenarios", type=int, help="Optional limit for smoke tests")
    parser.add_argument(
        "--no-human-penalty-filter",
        action="store_true",
        help="Disable the human-trajectory penalty filter (enabled by default)",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        return torch.device("cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {spec}")
    return device


def batch_features(features: Dict) -> Dict:
    """Add the batch dimension used by the original cached-data evaluator."""

    expected_dims = {
        "camera_feature": 3,
        "lidar_feature": 3,
        "status_feature": 1,
        "agent_trajectory": 3,
        "reward_gt": 2,
    }
    result = dict(features)
    for key, unbatched_dim in expected_dims.items():
        if key not in result:
            raise KeyError(f"Cached feature is missing required key '{key}'")
        if result[key].ndim == unbatched_dim:
            result[key] = result[key].unsqueeze(0)
        elif result[key].ndim != unbatched_dim + 1:
            raise ValueError(f"Unexpected {key} shape: {tuple(result[key].shape)}")
    return result


def cache_base_model(features: Dict) -> str:
    """Return the normalized base-model label from a candidate cache."""

    base_model = features.get("base_model")
    pipeline = features.get("pipeline")
    if base_model is None and isinstance(pipeline, str):
        base_model = pipeline.split("->", 1)[0]
    return str(base_model or "unknown")


def expected_pdm_scores(reward_gt: torch.Tensor) -> torch.Tensor:
    """Reconstruct EPDMS from the nine cached trajectory-reward components."""

    multipliers = reward_gt[:, 0:4, :].prod(dim=1, keepdim=True)
    weights = reward_gt.new_tensor([5 / 16, 5 / 16, 2 / 16, 2 / 16, 2 / 16]).view(1, 5, 1)
    weighted = (reward_gt[:, 4:9, :] * weights).sum(dim=1, keepdim=True)
    return multipliers * weighted


def score_trajectory(
    token: str,
    trajectory: Trajectory,
    metric_loader: MetricCacheLoader,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_policy: LogReplayTrafficAgents,
) -> pd.DataFrame:
    metric_cache = metric_loader.get_from_token(token)
    score_row, simulated_states = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=trajectory,
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_policy,
    )
    score_row["token"] = token
    score_row["valid"] = True
    score_row["log_name"] = metric_cache.log_name
    score_row["frame_type"] = metric_cache.scene_type
    score_row["start_time"] = metric_cache.timepoint.time_s
    end_pose = StateSE2(*trajectory.poses[-1])
    endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
    score_row["endpoint_x"] = endpoint.x
    score_row["endpoint_y"] = endpoint.y
    score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
    score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
    score_row["ego_simulated_states"] = [simulated_states]
    return score_row


def select_trajectory_batch(
    entries: Sequence[Tuple[str, str, Dict]],
    agent: RWMInferenceAgent,
) -> List[Dict[str, Any]]:
    """Rank one inference batch and return CPU-only tasks for parallel PDM scoring."""

    tensor_keys = (
        "camera_feature",
        "lidar_feature",
        "status_feature",
        "agent_trajectory",
        "reward_gt",
    )
    batched = {
        key: torch.cat([entry[2][key] for entry in entries], dim=0)
        for key in tensor_keys
    }
    predictions = agent.predict(batched)
    selected_indices = predictions["sim_logits"].argmax(dim=-1)[:, 0].detach().cpu()

    reward_gt = batched["reward_gt"].detach().cpu()
    epdms_scores = expected_pdm_scores(reward_gt)
    candidates = batched["agent_trajectory"].detach().cpu()

    tasks: List[Dict[str, Any]] = []
    for batch_idx, (token, base_model, _) in enumerate(entries):
        selected_idx = int(selected_indices[batch_idx].item())
        if selected_idx >= candidates.shape[1]:
            raise IndexError(
                f"RWM selected candidate {selected_idx}, but only {candidates.shape[1]} exist"
            )
        reward = reward_gt[batch_idx, :, selected_idx]
        row = {
            "token": token,
            "base_model": base_model,
            "valid": True,
            "no_at_fault_collisions": float(reward[0]),
            "drivable_area_compliance": float(reward[1]),
            "driving_direction_compliance": float(reward[2]),
            "traffic_light_compliance": float(reward[3]),
            "ego_progress": float(reward[4]),
            "time_to_collision_within_bound": float(reward[5]),
            "lane_keeping": float(reward[6]),
            "history_comfort": float(reward[7]),
            "comfort": np.nan,
            "two_frame_extended_comfort": float(reward[8]),
            "EPDMS": float(epdms_scores[batch_idx, 0, selected_idx]),
            "PDMS": np.nan,
        }
        tasks.append(
            {
                "token": token,
                "selected_index": selected_idx,
                "trajectory": candidates[batch_idx, selected_idx].numpy(),
                "row": row,
            }
        )
    return tasks


def score_selected_trajectory_batch(tasks: List[Dict[str, Any]]) -> List[Dict]:
    """NAVSIM worker target: PDM-score a chunk of already selected trajectories."""

    if not tasks:
        return []
    settings = tasks[0]["settings"]
    metric_loader = MetricCacheLoader(Path(settings["metric_cache"]))
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    model_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    simulator = PDMSimulator(proposal_sampling)
    scorer = PDMScorer(
        proposal_sampling,
        PDMScorerConfig(human_penalty_filter=settings["human_penalty_filter"]),
    )
    traffic_policy = LogReplayTrafficAgents(proposal_sampling)

    rows: List[Dict] = []
    for task in tasks:
        token = task["token"]
        row = dict(task["row"])
        try:
            trajectory = Trajectory(task["trajectory"], model_sampling)
            actual = score_trajectory(
                token,
                trajectory,
                metric_loader,
                simulator,
                scorer,
                traffic_policy,
            )
            row["comfort"] = float(actual["comfort"].iloc[0])
            row["PDMS"] = row["no_at_fault_collisions"] * row["drivable_area_compliance"] * (
                5 * row["ego_progress"]
                + 5 * row["time_to_collision_within_bound"]
                + 2 * row["comfort"]
            ) / 12
            rows.append(row)
        except Exception:
            LOGGER.error("PDM scoring failed for token %s\n%s", token, traceback.format_exc())
            if settings["fail_fast"]:
                raise
            rows.append(
                failed_result(
                    token,
                    base_model=row["base_model"],
                )
            )
    return rows


def build_evaluation_worker(args: argparse.Namespace) -> WorkerPool:
    """Construct the requested backend using NAVSIM/nuPlan's WorkerPool API."""

    if args.worker == "sequential":
        return Sequential()
    if args.worker in ("thread", "process"):
        return SingleMachineParallelExecutor(
            use_process_pool=args.worker == "process",
            max_workers=args.workers,
        )

    try:
        from nuplan.planning.utils.multithreading.worker_ray import RayDistributed
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ray backend requested but Ray is not installed; run "
            "`python -m pip install -e '.[distributed]'`"
        ) from exc
    return RayDistributed(
        threads_per_node=args.workers,
        output_dir=args.output_dir,
        use_distributed=args.ray_distributed,
    )


def failed_result(token: str, base_model: str = "unknown") -> Dict:
    row = asdict(PDMResults.get_empty_results())
    row.update(
        token=token,
        base_model=base_model,
        valid=False,
        two_frame_extended_comfort=np.nan,
        EPDMS=np.nan,
        PDMS=np.nan,
    )
    return row


def gather_torchrun_rows(
    local_rows: Sequence[Dict],
    num_shards: int,
    shard_index: int,
) -> Tuple[List[Dict], bool, bool]:
    """Gather torchrun shards on rank 0 after local CPU work has completed.

    Returns collected rows, whether this rank should write output, and whether
    the returned rows are the complete distributed result.
    """

    torchrun_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torchrun_rank = int(os.environ.get("RANK", "0"))
    if torchrun_world_size <= 1:
        return list(local_rows), True, False
    if num_shards != torchrun_world_size or shard_index != torchrun_rank:
        raise ValueError(
            "Under torchrun, --num-shards/--shard-index must match "
            f"WORLD_SIZE/RANK; got {num_shards}/{shard_index} versus "
            f"{torchrun_world_size}/{torchrun_rank}"
        )

    import torch.distributed as dist

    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend="gloo")
    gathered = [None] * torchrun_world_size if torchrun_rank == 0 else None
    try:
        dist.gather_object(list(local_rows), gathered, dst=0)
    finally:
        if initialized_here:
            dist.destroy_process_group()

    if torchrun_rank != 0:
        return [], False, True
    rows = [row for shard_rows in gathered for row in shard_rows]
    return rows, True, True


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row report containing counts and valid-scenario means."""

    valid = results["valid"].fillna(False).astype(bool)
    valid_results = results.loc[valid]
    summary: Dict[str, Any] = {
        "total_scenarios": len(results),
        "valid_scenarios": int(valid.sum()),
        "failed_scenarios": int((~valid).sum()),
        "success_rate": float(valid.mean()) if len(results) else np.nan,
        "base_model": "|".join(sorted(results["base_model"].dropna().astype(str).unique())),
    }
    for column in MEAN_METRIC_COLUMNS:
        values = pd.to_numeric(valid_results[column], errors="coerce")
        summary[f"mean_{column}"] = float(values.mean())
    return pd.DataFrame([summary])


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.workers < 1:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    if args.inference_batch_size < 1:
        raise ValueError(
            f"--inference-batch-size must be positive, got {args.inference_batch_size}"
        )
    num_shards = args.num_shards or int(os.environ.get("WORLD_SIZE", "1"))
    shard_index = (
        args.shard_index
        if args.shard_index is not None
        else int(os.environ.get("RANK", "0"))
    )
    if num_shards < 1:
        raise ValueError(f"--num-shards must be positive, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {num_shards}), got {shard_index}"
        )
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise ValueError(f"--torch-threads must be positive, got {args.torch_threads}")
        torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    LOGGER.info(
        "Using device=%s shard=%d/%d torch_threads=%d",
        device,
        shard_index,
        num_shards,
        torch.get_num_threads(),
    )

    metric_loader = MetricCacheLoader(args.metric_cache)
    feature_loader = FeatureCacheLoader(args.feature_cache)
    requested = load_tokens_csv(args.tokens_csv, args.token_column)
    if requested is None:
        requested = sorted(feature_loader.tokens)

    metric_tokens = set(metric_loader.tokens)
    feature_tokens = set(feature_loader.tokens)
    tokens = [token for token in requested if token in metric_tokens and token in feature_tokens]
    missing_metric = len([token for token in requested if token not in metric_tokens])
    missing_feature = len([token for token in requested if token not in feature_tokens])
    if missing_metric:
        LOGGER.warning("Skipping %d tokens without metric cache", missing_metric)
    if missing_feature:
        LOGGER.warning("Skipping %d tokens without feature cache", missing_feature)
    if args.max_scenarios is not None:
        tokens = tokens[: args.max_scenarios]
    if not tokens:
        raise RuntimeError("No tokens remain after intersecting token, metric-cache, and feature-cache inputs")
    all_tokens = tokens
    tokens = all_tokens[shard_index::num_shards]
    if not tokens:
        LOGGER.warning("Shard %d/%d has no tokens; waiting for result aggregation", shard_index, num_shards)

    if tokens:
        LOGGER.info(
            "Starting RWM selection for %d scenarios with batch_size=%d",
            len(tokens),
            args.inference_batch_size,
        )
    agent = RWMInferenceAgent(args.checkpoint, device) if tokens else None

    rows_by_token: Dict[str, Dict] = {}
    scoring_tasks: List[Dict[str, Any]] = []
    warned_unknown_provenance = False
    for batch_start in range(0, len(tokens), args.inference_batch_size):
        batch_tokens = tokens[batch_start : batch_start + args.inference_batch_size]
        entries: List[Tuple[str, str, Dict]] = []
        for token in batch_tokens:
            base_model = "unknown"
            try:
                features = feature_loader.load(token)
                base_model = cache_base_model(features)
                if args.expected_base_model is not None:
                    if base_model == "unknown" and not warned_unknown_provenance:
                        LOGGER.warning(
                            "Feature cache has no base-model metadata; cannot verify expected '%s'",
                            args.expected_base_model,
                        )
                        warned_unknown_provenance = True
                    elif base_model != "unknown" and base_model != args.expected_base_model:
                        raise ValueError(
                            f"Feature cache declares base_model={base_model!r}, "
                            f"expected {args.expected_base_model!r}"
                        )
                entries.append((token, base_model, batch_features(features)))
            except Exception:
                LOGGER.error("Feature loading failed for token %s\n%s", token, traceback.format_exc())
                if args.fail_fast:
                    raise
                rows_by_token[token] = failed_result(token, base_model)

        if not entries:
            continue
        try:
            selected_tasks = select_trajectory_batch(entries, agent)
        except Exception:
            if args.fail_fast:
                raise
            LOGGER.error(
                "Batched RWM selection failed; retrying %d scenes one by one\n%s",
                len(entries),
                traceback.format_exc(),
            )
            selected_tasks = []
            for entry in entries:
                try:
                    selected_tasks.extend(select_trajectory_batch([entry], agent))
                except Exception:
                    token, base_model, _ = entry
                    LOGGER.error("RWM selection failed for token %s\n%s", token, traceback.format_exc())
                    rows_by_token[token] = failed_result(token, base_model)

        settings = {
            "metric_cache": str(args.metric_cache.expanduser().resolve()),
            "human_penalty_filter": not args.no_human_penalty_filter,
            "fail_fast": args.fail_fast,
        }
        for task in selected_tasks:
            task["settings"] = settings
            scoring_tasks.append(task)
            row = task["row"]
            LOGGER.info(
                "RWM selected token=%s EPDMS=%.4f",
                task["token"],
                row["EPDMS"],
            )

    del agent
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if scoring_tasks:
        worker = build_evaluation_worker(args)
        LOGGER.info(
            "Starting PDM scoring for %d scenarios with worker=%s workers=%d",
            len(scoring_tasks),
            args.worker,
            worker.number_of_threads,
        )
        scored_rows = worker_map(worker, score_selected_trajectory_batch, scoring_tasks)
        rows_by_token.update({row["token"]: row for row in scored_rows})
        shutdown = getattr(worker, "shutdown", None)
        if callable(shutdown):
            shutdown()

    local_rows = [rows_by_token[token] for token in tokens]
    collected_rows, should_write, aggregated = gather_torchrun_rows(
        local_rows,
        num_shards,
        shard_index,
    )
    if not should_write:
        LOGGER.info("Shard %d/%d sent %d rows to rank 0", shard_index, num_shards, len(local_rows))
        return

    output_tokens = all_tokens if aggregated else tokens
    collected_by_token = {row["token"]: row for row in collected_rows}
    missing_rows = [token for token in output_tokens if token not in collected_by_token]
    if missing_rows:
        raise RuntimeError(f"Missing {len(missing_rows)} rows after result aggregation")
    rows = [collected_by_token[token] for token in output_tokens]
    results = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = f"{datetime.now():%Y.%m.%d.%H.%M.%S}"
    rank_suffix = (
        f"_rank{shard_index:03d}-of-{num_shards:03d}"
        if num_shards > 1 and not aggregated
        else ""
    )
    output_path = args.output_dir / f"rwm_pdm_{timestamp}{rank_suffix}.csv"
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    results.to_csv(output_path, index=False)
    summary = summarize_results(results)
    summary.to_csv(summary_path, index=False)
    valid = results["valid"].fillna(False).astype(bool)
    LOGGER.info(
        "Finished: %d succeeded, %d failed, mean EPDMS=%.6f, mean PDMS=%.6f, "
        "output=%s, summary=%s",
        int(valid.sum()),
        int((~valid).sum()),
        results.loc[valid, "EPDMS"].mean(),
        results.loc[valid, "PDMS"].mean(),
        output_path,
        summary_path,
    )


if __name__ == "__main__":
    main()
