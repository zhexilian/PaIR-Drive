"""Sensor preprocessing shared by DiffusionDrive, Parallel-IR, and RWM."""

from typing import Dict

import cv2
import numpy as np
import torch

from navsim.common.dataclasses import AgentInput
from navsim.common.enums import LidarIndex
from pairdrive.models.diffusiondrive import DiffusionDriveConfig


def build_sensor_features(
    agent_input: AgentInput, config: DiffusionDriveConfig = DiffusionDriveConfig()
) -> Dict[str, torch.Tensor]:
    cameras = agent_input.cameras[-1]
    left = cameras.cam_l0.image[28:-28, 416:-416]
    front = cameras.cam_f0.image[28:-28]
    right = cameras.cam_r0.image[28:-28, 416:-416]
    stitched = np.concatenate([left, front, right], axis=1)
    resized = cv2.resize(stitched, (config.camera_width, config.camera_height))
    camera_feature = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)

    point_cloud = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T
    point_cloud = point_cloud[point_cloud[..., 2] < config.max_height_lidar]

    xbins = np.linspace(
        config.lidar_min_x,
        config.lidar_max_x,
        int((config.lidar_max_x - config.lidar_min_x) * config.pixels_per_meter) + 1,
    )
    ybins = np.linspace(
        config.lidar_min_y,
        config.lidar_max_y,
        int((config.lidar_max_y - config.lidar_min_y) * config.pixels_per_meter) + 1,
    )

    def splat(points: np.ndarray) -> np.ndarray:
        histogram = np.histogramdd(points[:, :2], bins=(xbins, ybins))[0]
        return np.minimum(histogram, config.hist_max_per_pixel) / config.hist_max_per_pixel

    below = point_cloud[point_cloud[..., 2] <= config.lidar_split_height]
    above = point_cloud[point_cloud[..., 2] > config.lidar_split_height]
    channels = [splat(above)]
    if config.use_ground_plane:
        channels.insert(0, splat(below))
    lidar_feature = torch.from_numpy(np.stack(channels).astype(np.float32))

    ego = agent_input.ego_statuses[-1]
    status_feature = torch.cat(
        [
            torch.as_tensor(ego.driving_command, dtype=torch.float32),
            torch.as_tensor(ego.ego_velocity, dtype=torch.float32),
            torch.as_tensor(ego.ego_acceleration, dtype=torch.float32),
        ]
    )
    return {
        "camera_feature": camera_feature,
        "lidar_feature": lidar_feature,
        "status_feature": status_feature,
    }

