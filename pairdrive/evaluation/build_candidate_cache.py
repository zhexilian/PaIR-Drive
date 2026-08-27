"""Build DiffusionDrive -> Parallel-IR candidate caches for RWM evaluation."""

import argparse
import gzip
import logging
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import (
    SceneAggregator,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
from pairdrive.data import MetricCacheLoader, load_tokens_csv
from pairdrive.evaluation.run_rwm_pdm_score import resolve_device, score_trajectory
from pairdrive.features import build_sensor_features
from pairdrive.models.trajectory_pipeline import CandidateGenerator


LOGGER = logging.getLogger("pairdrive.candidate_cache")
REWARD_COLUMNS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "two_frame_extended_comfort",
]
_WORKER_METRIC_LOADER: Optional[MetricCacheLoader] = None
_WORKER_SIMULATOR: Optional[PDMSimulator] = None
_WORKER_SCORER: Optional[PDMScorer] = None
_WORKER_TRAFFIC_POLICY: Optional[LogReplayTrafficAgents] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-path", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--parallel-ir-checkpoint", type=Path, required=True)
    parser.add_argument("--tokens-csv", type=Path)
    parser.add_argument("--token-column", default="token")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--scoring-worker",
        choices=("sequential", "process"),
        default="process",
        help="Backend for per-candidate NAVSIM PDM scoring",
    )
    parser.add_argument("--workers", type=int, default=16, help="PDM scoring processes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _scene_loader(args: argparse.Namespace, tokens: Optional[List[str]]) -> SceneLoader:
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        tokens=tokens,
        max_scenes=args.max_scenarios,
    )
    current_frame = [3]
    sensors = SensorConfig(
        cam_f0=current_frame,
        cam_l0=current_frame,
        cam_l1=False,
        cam_l2=False,
        cam_r0=current_frame,
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=current_frame,
    )
    return SceneLoader(
        data_path=args.navsim_log_path.expanduser().resolve(),
        original_sensor_path=args.sensor_path.expanduser().resolve(),
        scene_filter=scene_filter,
        sensor_config=sensors,
    )


def _previous_token_map(
    tokens: List[str], scene_loader: SceneLoader
) -> Dict[str, str]:
    """Build adjacent-frame links from already-loaded raw log metadata."""

    by_log: Dict[str, List[tuple]] = {}
    for token in tokens:
        frames = scene_loader.scene_frames_dicts[token]
        current_frame = next(frame for frame in frames if frame["token"] == token)
        time_s = float(current_frame["timestamp"]) / 1e6
        by_log.setdefault(current_frame["log_name"], []).append((time_s, token))

    previous: Dict[str, str] = {}
    for entries in by_log.values():
        entries.sort()
        for (previous_time, previous_token), (now_time, now_token) in zip(entries, entries[1:]):
            if 0 < now_time - previous_time < 0.55:
                previous[now_token] = previous_token
    return previous


def _reward_vector(
    token: str,
    trajectory: Trajectory,
    previous_token: Optional[str],
    previous_score: Optional[pd.DataFrame],
    metric_loader: MetricCacheLoader,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_policy: LogReplayTrafficAgents,
) -> torch.Tensor:
    current = score_trajectory(
        token, trajectory, metric_loader, simulator, scorer, traffic_policy
    )
    two_frame_comfort = np.nan
    if previous_token is not None and previous_score is not None:
        paired = pd.concat([current, previous_score], ignore_index=True).set_index("token")
        update = SceneAggregator(
            now_frame=token,
            previous_frame=previous_token,
            score_df=paired,
            proposal_sampling=simulator.proposal_sampling,
        ).aggregate_scores(one_stage_only=True)
        two_frame_comfort = float(update.iloc[0]["two_frame_extended_comfort"])

    values = [float(current.iloc[0][column]) for column in REWARD_COLUMNS[:-1]]
    values.append(two_frame_comfort)
    return torch.nan_to_num(torch.tensor(values, dtype=torch.float32), nan=0.0)


def _initialize_scoring_worker(metric_cache_path: str) -> None:
    """Initialize persistent NAVSIM scoring objects in one spawned process."""

    global _WORKER_METRIC_LOADER
    global _WORKER_SIMULATOR
    global _WORKER_SCORER
    global _WORKER_TRAFFIC_POLICY

    torch.set_num_threads(1)
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    _WORKER_METRIC_LOADER = MetricCacheLoader(Path(metric_cache_path))
    _WORKER_SIMULATOR = PDMSimulator(proposal_sampling)
    _WORKER_SCORER = PDMScorer(
        proposal_sampling,
        PDMScorerConfig(human_penalty_filter=True),
    )
    _WORKER_TRAFFIC_POLICY = LogReplayTrafficAgents(proposal_sampling)


def _score_candidate_task(
    task: Tuple[str, np.ndarray, Optional[str], Optional[pd.DataFrame]],
) -> torch.Tensor:
    """Score one candidate in a persistent CPU process."""

    if any(
        component is None
        for component in (
            _WORKER_METRIC_LOADER,
            _WORKER_SIMULATOR,
            _WORKER_SCORER,
            _WORKER_TRAFFIC_POLICY,
        )
    ):
        raise RuntimeError("Candidate scoring worker was not initialized")
    token, poses, previous_token, previous_score = task
    return _reward_vector(
        token,
        Trajectory(
            poses,
            TrajectorySampling(time_horizon=4, interval_length=0.5),
        ),
        previous_token,
        previous_score,
        _WORKER_METRIC_LOADER,
        _WORKER_SIMULATOR,
        _WORKER_SCORER,
        _WORKER_TRAFFIC_POLICY,
    )


def _save_feature(path: Path, feature: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as stream:
        pickle.dump(feature, stream, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.workers < 1:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    device = resolve_device(args.device)

    requested = load_tokens_csv(args.tokens_csv, args.token_column)
    scene_loader = _scene_loader(args, requested)
    metric_loader = MetricCacheLoader(args.metric_cache)
    metric_tokens = set(metric_loader.tokens)
    tokens = [token for token in scene_loader.tokens if token in metric_tokens]
    if args.max_scenarios is not None:
        tokens = tokens[: args.max_scenarios]
    if not tokens:
        raise RuntimeError("No raw scenes have a matching metric cache")

    LOGGER.info("Building previous-frame map for %d scenes from loaded log metadata", len(tokens))
    previous_tokens = _previous_token_map(tokens, scene_loader)
    LOGGER.info("Loading DiffusionDrive and Parallel-IR checkpoints on %s", device)
    generator = CandidateGenerator(
        args.base_checkpoint, args.parallel_ir_checkpoint, device
    )
    LOGGER.info("DiffusionDrive and Parallel-IR checkpoints loaded")
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling)
    scorer = PDMScorer(proposal_sampling, PDMScorerConfig(human_penalty_filter=True))
    traffic_policy = LogReplayTrafficAgents(proposal_sampling)
    output_root = args.output_dir.expanduser().resolve()
    scoring_pool = None
    if args.scoring_worker == "process":
        scoring_workers = min(args.workers, 16)
        if args.workers > scoring_workers:
            LOGGER.warning(
                "Capping PDM scoring processes at %d because each scene has only 16 candidates",
                scoring_workers,
            )
        scoring_pool = ProcessPoolExecutor(
            max_workers=scoring_workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_scoring_worker,
            initargs=(str(args.metric_cache.expanduser().resolve()),),
        )
        LOGGER.info(
            "Using %d spawned CPU processes for candidate PDM scoring",
            scoring_workers,
        )

    successes = 0
    failures = 0
    try:
        for index, token in enumerate(tokens, start=1):
            metric_cache = metric_loader.get_from_token(token)
            output_path = output_root / metric_cache.log_name / token / "pairdrive_feature.gz"
            if output_path.is_file() and not args.force:
                LOGGER.info("Skipping existing %d/%d: %s", index, len(tokens), token)
                continue
            LOGGER.info(
                "Building DiffusionDrive -> Parallel-IR candidates %d/%d: %s",
                index,
                len(tokens),
                token,
            )
            try:
                scene = scene_loader.get_scene_from_token(token)
                feature = build_sensor_features(scene.get_agent_input())
                generated = generator.generate(feature)
                feature.update(generated)
                feature["expert_feature"] = torch.as_tensor(
                    scene.get_future_trajectory(num_trajectory_frames=8).poses
                )
                feature["token"] = token

                previous_token = previous_tokens.get(token)
                previous_score = None
                if previous_token is not None:
                    previous_scene = scene_loader.get_scene_from_token(previous_token)
                    human_trajectory = previous_scene.get_future_trajectory(
                        num_trajectory_frames=8
                    )
                    previous_score = score_trajectory(
                        previous_token,
                        human_trajectory,
                        metric_loader,
                        simulator,
                        scorer,
                        traffic_policy,
                    )

                if scoring_pool is not None:
                    tasks = [
                        (token, poses.numpy(), previous_token, previous_score)
                        for poses in generated["agent_trajectory"]
                    ]
                    rewards = list(
                        scoring_pool.map(_score_candidate_task, tasks, chunksize=1)
                    )
                else:
                    rewards = [
                        _reward_vector(
                            token,
                            Trajectory(
                                poses.numpy(),
                                TrajectorySampling(time_horizon=4, interval_length=0.5),
                            ),
                            previous_token,
                            previous_score,
                            metric_loader,
                            simulator,
                            scorer,
                            traffic_policy,
                        )
                        for poses in generated["agent_trajectory"]
                    ]
                feature["reward_gt"] = torch.stack(rewards, dim=1)
                _save_feature(output_path, feature)
                successes += 1
            except Exception:
                failures += 1
                LOGGER.error("Candidate caching failed for %s\n%s", token, traceback.format_exc())
                if args.fail_fast:
                    raise
    finally:
        if scoring_pool is not None:
            scoring_pool.shutdown(wait=True, cancel_futures=True)

    LOGGER.info("Candidate caching complete: %d succeeded, %d failed", successes, failures)


if __name__ == "__main__":
    main()
