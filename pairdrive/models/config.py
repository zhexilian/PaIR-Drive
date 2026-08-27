"""Inference-only model configuration for Pair-Drive."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RWMConfig:
    """Architecture values required to reconstruct the released RWM."""

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    local_weight_path: Optional[str] = None

    latent: bool = False
    use_ground_plane: bool = False
    lidar_seq_len: int = 1

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
    num_bounding_boxes: int = 30

    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    num_keyval: int = 64
    num_fut_timestep: int = 1
