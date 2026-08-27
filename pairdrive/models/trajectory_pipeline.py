"""Base trajectory inference followed by Parallel-IR expansion."""

from pathlib import Path
from typing import Dict, Mapping

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from pairdrive.models.diffusiondrive import DiffusionDriveConfig, V2TransfuserModel
from pairdrive.models.parallel_ir import ParallelIRConfig, ParallelIRModel


def _checkpoint_model_state(
    checkpoint_path: Path, model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state: Mapping[str, torch.Tensor] = checkpoint.get("state_dict", checkpoint)
    prefixes = ("agent._transfuser_model.", "_transfuser_model.")
    mapped: Dict[str, torch.Tensor] = {}
    model_keys = set(model.state_dict())
    for key, value in state.items():
        mapped_key = key
        for prefix in prefixes:
            if key.startswith(prefix):
                mapped_key = key[len(prefix) :]
                break
        if mapped_key in model_keys:
            mapped[mapped_key] = value

    missing = model_keys - set(mapped)
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} misses {len(missing)} model tensors; examples: {sample}"
        )
    return mapped


class CandidateGenerator:
    """Generate one base trajectory and fifteen Parallel-IR candidates."""

    def __init__(
        self,
        base_checkpoint: Path,
        parallel_ir_checkpoint: Path,
        device: torch.device,
    ):
        self.device = device
        self.base_model_name = "diffusiondrive"
        self.base_model = V2TransfuserModel(DiffusionDriveConfig())

        self.parallel_ir = ParallelIRModel(
            TrajectorySampling(time_horizon=4, interval_length=0.5),
            ParallelIRConfig(),
        )
        self.base_model.load_state_dict(
            _checkpoint_model_state(base_checkpoint, self.base_model), strict=True
        )
        self.parallel_ir.load_state_dict(
            _checkpoint_model_state(parallel_ir_checkpoint, self.parallel_ir), strict=True
        )
        self.base_model.to(device).eval()
        self.parallel_ir.to(device).eval()

    def generate(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model_features = {
            key: features[key].unsqueeze(0).to(self.device, non_blocking=True)
            for key in ("camera_feature", "lidar_feature", "status_feature")
        }
        with torch.inference_mode():
            base_output = self.base_model(model_features)
            base_trajectory = base_output["best_trajectory"]
            expanded = self.parallel_ir(
                {**model_features, "expert_feature": base_trajectory}
            )["trajectory"]

        codes = sorted(expanded)
        if len(codes) != 15:
            raise RuntimeError(f"Parallel-IR returned {len(codes)} candidates, expected 15")
        candidates = torch.stack([expanded[code] for code in codes], dim=1)
        candidates = torch.cat([candidates, base_trajectory[:, None]], dim=1)
        return {
            "base_trajectory": base_trajectory[0].float().cpu(),
            "agent_trajectory": candidates[0].float().cpu(),
            "candidate_codes": codes + [f"{self.base_model_name}_base"],
            "base_model": self.base_model_name,
        }
