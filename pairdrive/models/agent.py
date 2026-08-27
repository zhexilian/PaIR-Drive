"""Small inference wrapper around the RWM architecture."""

from pathlib import Path
from typing import Dict, Mapping

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from pairdrive.models.config import RWMConfig
from pairdrive.models.rwm import RWMModel


class RWMInferenceAgent:
    """Load an RWM checkpoint and rank precomputed trajectory candidates."""

    def __init__(self, checkpoint_path: Path, device: torch.device):
        self.trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
        self.device = device
        self.model = RWMModel(self.trajectory_sampling, RWMConfig())
        self._load_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _extract_model_state(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prefixes = (
            "agent._transfuser_model.",
            "agent.WoTE_model.",
            "_transfuser_model.",
            "WoTE_model.",
        )
        mapped: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            model_key = key
            for prefix in prefixes:
                if key.startswith(prefix):
                    model_key = key[len(prefix) :]
                    break
            mapped[model_key] = value
        return mapped

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"RWM checkpoint does not exist: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        mapped = self._extract_model_state(state_dict)
        self.model.load_state_dict(mapped, strict=True)

    def predict(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        required = ("camera_feature", "lidar_feature", "status_feature", "agent_trajectory")
        missing = [key for key in required if key not in features]
        if missing:
            raise KeyError(f"Cached feature is missing keys: {missing}")

        model_features = {
            key: features[key].to(self.device, non_blocking=True)
            for key in ("camera_feature", "lidar_feature", "status_feature")
        }
        candidates = features["agent_trajectory"].to(self.device, non_blocking=True)
        with torch.inference_mode():
            return self.model(model_features, candidates)
