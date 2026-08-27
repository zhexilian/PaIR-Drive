"""Architecture constants required by DiffusionDrive inference."""

from dataclasses import dataclass, field
from pathlib import Path

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class DiffusionDriveConfig:
    trajectory_sampling: TrajectorySampling = field(
        default_factory=lambda: TrajectorySampling(time_horizon=4, interval_length=0.5)
    )
    plan_anchor_path: str = str(
        Path(__file__).resolve().parents[2] / "assets" / "kmeans_navsim_traj_20.npy"
    )

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    latent: bool = False
    lidar_seq_len: int = 1
    use_ground_plane: bool = False

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32
    max_height_lidar: float = 100.0
    lidar_split_height: float = 0.2
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256
    img_vert_anchors: int = 8
    img_horz_anchors: int = 32
    lidar_vert_anchors: int = 8
    lidar_horz_anchors: int = 8

    block_exp: int = 4
    n_layer: int = 2
    n_head: int = 4
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    gpt_linear_layer_init_mean: float = 0.0
    gpt_linear_layer_init_std: float = 0.02
    gpt_layer_norm_init_weight: float = 1.0

    perspective_downsample_factor: int = 1
    transformer_decoder_join: bool = True
    detect_boxes: bool = True
    use_bev_semantic: bool = True
    use_semantic: bool = False
    use_depth: bool = False
    add_features: bool = True

    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0
    num_bounding_boxes: int = 30
    num_bev_classes: int = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

