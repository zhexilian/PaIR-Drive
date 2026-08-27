from typing import Dict

import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from pairdrive.models.config import RWMConfig
from pairdrive.models.transfuser_backbone import TransfuserBackbone


class RWMModel(nn.Module):
    """Reward/world model used to rank cached trajectory candidates."""

    def __init__(self, trajectory_sampling: TrajectorySampling, config: RWMConfig):
        """
        Initializes TransFuser torch module.
        :param trajectory_sampling: trajectory sampling specification.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._bev_embedding = nn.Embedding(8**2, config.tf_d_model)  # 8x8 feature grid
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self.mlp_planning_vb = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        # Transformer Encoder for trajectory_anchors_feat
        TRANSFORMER_DIM_FEEDFORWARD = 512
        TRANSFORMER_NHEAD = 8
        TRANSFORMER_DROPOUT = 0.1
        TRANSFORMER_NUM_LAYERS = 2
        hidden_dim = 256

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=TRANSFORMER_NHEAD, 
            dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD, 
            dropout=TRANSFORMER_DROPOUT,
            batch_first=True
        )
        self.cluster_encoder = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_NUM_LAYERS)
        self.encode_ego_feat_mlp = nn.Sequential(
            nn.Linear(2 * 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        # NOTE latent world model
        num_keyval = config.num_keyval if hasattr(config, 'num_keyval') else 64
        self.num_plan_queries = num_keyval
        self.num_scenes = self.num_plan_queries + 1  # including the ego feat and action
        
        self.scene_position_embedding = nn.Embedding(self.num_scenes, hidden_dim)

        self.encode_ego_feat_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        wm_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=TRANSFORMER_NHEAD, 
            dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD, 
            dropout=TRANSFORMER_DROPOUT,
            batch_first=True
        )
        self.latent_world_model = nn.TransformerEncoder(wm_encoder_layer, num_layers=TRANSFORMER_NUM_LAYERS)

        # reward decoder
        self.num_fut_timestep = config.num_fut_timestep if hasattr(config, 'num_fut_timestep') else 1
        self.reward_conv_net = RewardConvNet(input_channels=(self.num_fut_timestep+1) * hidden_dim, conv1_out_channels=hidden_dim, conv2_out_channels=hidden_dim)
        self.reward_cat_head = nn.Sequential(
            nn.Linear((self.num_fut_timestep + 1 + 1) * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        NUM_SCORE_HEADS = 9
        self.reward_trunk = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.reward_out = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            for _ in range(NUM_SCORE_HEADS)
        ])

        self.reward_logits = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

    def _get_trajectory_anchors_feat(self, trajectory_anchors, batch_size: int):
        """
        Get the features of the cluster centers and expand them to the batch size.
        """
        num_traj = trajectory_anchors.shape[1]
        device = trajectory_anchors.device
        init_traj = trajectory_anchors.reshape(batch_size, num_traj, -1).to(device)  # [bz, num_traj, 24]
        trajectory_anchors_feat = self.mlp_planning_vb(init_traj)  # [bz, num_traj, hidden_dim256]
        trajectory_anchors_feat = self.cluster_encoder(trajectory_anchors_feat)  # [bz, num_traj, encoded_dim]
        return trajectory_anchors_feat, num_traj
    
    def _concatenate_ego_and_traj_features(self, ego_status_feat: torch.Tensor, trajectory_anchors_feat: torch.Tensor) -> torch.Tensor:
        """
        Concatenate ego features with trajectory features and encode.
        """
        # Repeat ego_status_feat to match the number of trajectories
        ego_status_feat = ego_status_feat.repeat(1, trajectory_anchors_feat.shape[1], 1)  # [batch_size, num_traj, C]
        # Concatenate
        ego_feat = torch.cat([ego_status_feat, trajectory_anchors_feat], dim=-1)  # [batch_size, num_traj, C + encoded_dim]
        # Encode
        ego_feat = self.encode_ego_feat_mlp(ego_feat)  # [batch_size, num_traj, C']
        ego_feat = ego_feat.unsqueeze(-2)  # [batch_size, num_traj, 1, C']
        return ego_feat

    def encode_traj_into_ego_feat(self, ego_status_feat: torch.Tensor, init_trajectory_anchor: torch.Tensor, batch_size: int):
        """
        Encode trajectory into ego feature by processing cluster centers and concatenating features.

        """
        trajectory_anchors_feat, num_traj = self._get_trajectory_anchors_feat(init_trajectory_anchor, batch_size)
        ego_feat_encoded = self._concatenate_ego_and_traj_features(ego_status_feat, trajectory_anchors_feat)
        return ego_feat_encoded, num_traj

    def inject_ego_feat_to_bev_map(self, bev_map, new_features, delta_x_y, H=8, W=8):
        """`
        Add a new feature vector in batch to the corresponding location in the BEV feature map, affecting the four pixels around each position.

        Parameters:
        - bev_map (torch.Tensor): BEV feature map with shape (B, C, H, W)
        - delta_x_y (torch.Tensor): x and y coordinates in the ego coordinate system, shape (B, 2)
        - new_features (torch.Tensor): Feature vectors to be added, shape (B, C)
        - H (int): Height (in pixels) of the BEV feature map
        - W (int): Width (in pixels) of the BEV feature map

        Returns:
        - updated_bev_map (torch.Tensor): Updated BEV feature map, shape (B, C, H, W)
        """
        B, C, H_map, W_map = bev_map.shape
        assert H_map == H and W_map == W, f"BEV map dimensions must be ({H}, {W}), but got ({H_map}, {W_map})"
        assert new_features.shape == (B, C), "new_features must have shape (B, C)"

        device = bev_map.device
        dtype = bev_map.dtype

        delta_x, delta_y = delta_x_y[:, 0], delta_x_y[:, 1]
        # Calculate the pixel-to-meter ratio
        pixel_per_meter_x = H / 32.0  # 32 meters covered by H pixels
        pixel_per_meter_y = W / 64.0  # 64 meters covered by W pixels

        # Convert ego coordinates to floating point pixel indices
        h_idx = delta_x * pixel_per_meter_x  # (B,)
        w_idx = delta_y * pixel_per_meter_y + (W / 2.0)  # Origin at (0, W/2)

        # Get four nearby integer pixel indices
        h0 = torch.floor(h_idx).long()  # (B,)
        w0 = torch.floor(w_idx).long()  # (B,)
        h1 = h0 + 1
        w1 = w0 + 1

        # Compute distance weights
        dh = h_idx - h0.float()  # (B,)
        dw = w_idx - w0.float()  # (B,)

        w00 = (1 - dh) * (1 - dw)  # (B,)
        w01 = (1 - dh) * dw
        w10 = dh * (1 - dw)
        w11 = dh * dw

        # Stack indices and weights for all four nearby pixels
        # Each point contributes to four positions
        h_indices = torch.stack([h0, h0, h1, h1], dim=1)  # (B, 4)
        w_indices = torch.stack([w0, w1, w0, w1], dim=1)  # (B, 4)
        weights = torch.stack([w00, w01, w10, w11], dim=1)  # (B, 4)

        # Create batch indices
        batch_indices = torch.arange(B, device=device).view(B, 1).repeat(1, 4)  # (B, 4)

        # Flatten contributions
        h_indices_flat = h_indices.reshape(-1)  # (B*4,)
        w_indices_flat = w_indices.reshape(-1)  # (B*4,)
        weights_flat = weights.reshape(-1)  # (B*4,)
        batch_indices_flat = batch_indices.reshape(-1)  # (B*4,)

        # Create validity mask to ensure indices are within range
        valid = (h_indices_flat >= 0) & (h_indices_flat < H) & (w_indices_flat >= 0) & (w_indices_flat < W)  # (B*4,)

        if not valid.any():
            print("No valid indices")
            return bev_map

        # Filter out invalid indices and weights
        h_indices_valid = h_indices_flat[valid]  # (M,)
        w_indices_valid = w_indices_flat[valid]  # (M,)
        weights_valid = weights_flat[valid].unsqueeze(1)  # (M, 1)
        batch_indices_valid = batch_indices_flat[valid]  # (M,)

        # Get corresponding feature vectors and weights
        # Repeat each new_feature four times corresponding to the four weights
        new_features_expanded = new_features[batch_indices_valid]  # (M, C)
        weighted_features = new_features_expanded * weights_valid  # (M, C)

        # Compute linear indices
        # linear_index = b * C * H * W + c * H * W + h * W + w
        # Compute indices for each channel using broadcasting
        c_indices = torch.arange(C, device=device).view(1, C).repeat(weighted_features.shape[0], 1)  # (M, C)
        linear_indices = (batch_indices_valid.unsqueeze(1) * C * H * W) + (c_indices * H * W) + (h_indices_valid.unsqueeze(1) * W) + w_indices_valid.unsqueeze(1)  # (M, C)
        linear_indices = linear_indices.reshape(-1)  # (M*C,)

        # Flatten weighted features to (M*C,)
        weighted_features_flat = weighted_features.reshape(-1)  # (M*C,)

        # Flatten bev_map to (B*C*H*W,)
        bev_map_flat = bev_map.reshape(-1)  # (B*C*H*W,)

        # Use index_add_ to accumulate weighted features at corresponding positions
        bev_map_flat.index_add_(0, linear_indices, weighted_features_flat)

        # Reshape the BEV feature map back to (B, C, H, W)
        updated_bev_map = bev_map_flat.view(B, C, H, W)

        return updated_bev_map

    def _inject_cur_ego_into_bev(self, scene_bev_feature: torch.Tensor, ego_feat: torch.Tensor, num_traj: int, h = 8, w = 8) -> torch.Tensor:
        """
        Inject the ego feature into the BEV map.
        """
        bz = scene_bev_feature.shape[0]
        scene_bev_feature = scene_bev_feature.permute(0, 1, 3, 2).reshape(bz*num_traj, -1, h, w)  # [batch_size, num_traj, C, H*W]
        ego_feat = ego_feat.squeeze(2).reshape(bz*num_traj, -1)  # [batch_size*num_traj, C]
         # [batch_size*num_traj, 2]
        coors = torch.zeros(bz*num_traj, 2).to(ego_feat.device)
        scene_bev_feature = self.inject_ego_feat_to_bev_map(scene_bev_feature, ego_feat, coors)
        scene_bev_feature = scene_bev_feature.view(bz, num_traj, -1, h * w)  # [batch_size, num_traj, C, H*W]
        scene_bev_feature = scene_bev_feature.permute(0, 1, 3, 2)  # [batch_size, num_traj, H*W, C]
        return scene_bev_feature
    
    def _compute_reward_feature(self, fut_ego_feat_list, fut_flatten_bev_feature_multi_trajs_list, batch_size, num_traj, h=8, w=8) -> torch.Tensor:
        """
        Compute the scoring features.
        """
        bev_feat_list = []
        for bev_feat in  fut_flatten_bev_feature_multi_trajs_list:
            bev_feat = bev_feat.view(batch_size * num_traj, h, w, -1) 
            bev_feat_list.append(bev_feat)
        all_bev_feature = torch.cat(bev_feat_list, dim=-1)  # [batch_size*num_traj, H, W, C1 + C2]
        all_bev_feature = all_bev_feature.permute(0, 3, 1, 2)  # [batch_size*num_traj, C1 + C2, H, W]
        
        # Apply convolution network
        reward_conv_output = self.reward_conv_net(all_bev_feature).squeeze(-1).permute(0, 2, 1)  # [batch_size*num_traj, 1, C_conv]
        
        # Prepare scoring features
        cat_reward_feature = torch.cat(fut_ego_feat_list + [reward_conv_output], dim=1)  # [batch_size*num_traj, 3, C_total]
        cat_reward_feature = cat_reward_feature.reshape(batch_size * num_traj, -1)  # [batch_size*num_traj, 3*C_total]
        
        # Apply scoring head
        reward_feature = self.reward_cat_head(cat_reward_feature)  # [batch_size*num_traj, 256]
        reward_feature = reward_feature.view(batch_size, num_traj, -1)  # [batch_size, num_traj, 256]
        return reward_feature
    
    def _latent_world_model_processing(self, flatten_bev_feature_multi_trajs: torch.Tensor, ego_feat: torch.Tensor, batch_size: int, num_traj: int) -> torch.Tensor:
        """
        Process features through the latent world model.
        """
        # Concatenate ego_feat and flatten_bev_feature_multi_trajs
        # Assume concatenation along the feature dimension
        scene_feature = torch.cat([ego_feat, flatten_bev_feature_multi_trajs], dim=1)  # [batch_size, num_traj, 1 + 32, C]
        
        # Add positional embedding
        scene_position_embedding = self.scene_position_embedding.weight[None,  :, :].repeat(batch_size * num_traj, 1, 1)  # [batch_size * num_traj, 1 + 32, embedding_dim]

        scene_feature = scene_feature + scene_position_embedding
        
        # Reshape to fit the latent world model
        fut_scene_feature = self.latent_world_model(scene_feature)  # [batch_size*num_traj, C_new, H'*W']
        fut_ego_feat = fut_scene_feature[:, 0:1]
        fut_flatten_bev_feature_multi_trajs = fut_scene_feature[:, 1:]
        return fut_ego_feat, fut_flatten_bev_feature_multi_trajs
    
    def _inject_fut_ego_into_bev(self, agent_trajectory, scene_bev_feature: torch.Tensor, ego_feat: torch.Tensor, num_traj: int, fut_idx = 8, h = 8, w = 8) -> torch.Tensor:
        """
        Inject the ego feature into the BEV map.
        """
        bz = int(scene_bev_feature.shape[0] // num_traj)
        scene_bev_feature = scene_bev_feature.permute(0, 2, 1).reshape(bz*num_traj, -1, h, w)  # [batch_size, num_traj, C, H*W]
        ego_feat = ego_feat.squeeze(2).reshape(bz*num_traj, -1)  # [batch_size*num_traj, C]
         # [batch_size*num_traj, 2]
        # coors = torch.zeros(bz*num_traj, 2).to(ego_feat.device)
        coors = agent_trajectory[:, :, fut_idx-1, :2].to(ego_feat.device).reshape(bz*num_traj, -1)

        scene_bev_feature = self.inject_ego_feat_to_bev_map(scene_bev_feature, ego_feat, coors)
        scene_bev_feature = scene_bev_feature.view(bz*num_traj, -1, h * w)  # [batch_size, num_traj, C, H*W]
        scene_bev_feature = scene_bev_feature.permute(0, 2, 1)  # [batch_size*num_traj, H*W, C]
        return scene_bev_feature

    def forward(self, features: Dict[str, torch.Tensor], agent_trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        num_traj = agent_trajectory.shape[1]
        batch_size = agent_trajectory.shape[0]

        camera_feature: torch.Tensor = features["camera_feature"]
        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        # flatten bev feature
        flatten_bev_feature = bev_feature + self._keyval_embedding.weight[None, :, :]
        # ego status feature
        status_encoding = self._status_encoding(status_feature)
        ego_status_feat = status_encoding[:, None, :]
        # print('flatten_bev_feature:', flatten_bev_feature.shape)
        # print("ego_status_feat:", ego_status_feat.shape)

        ego_feat, _ = self.encode_traj_into_ego_feat(ego_status_feat, agent_trajectory, batch_size) # [batch_size, num_traj, 1, C']
        flatten_bev_feature_multi_trajs = flatten_bev_feature.unsqueeze(1).repeat(1, num_traj, 1, 1) # [batch_size, num_traj, HW, C']
        # [batch_size, num_traj, H*W, C]
        flatten_bev_feature_multi_trajs = self._inject_cur_ego_into_bev(
            flatten_bev_feature_multi_trajs, ego_feat, num_traj)
        # print('flatten_bev_feature_nulti_trajs:', flatten_bev_feature_multi_trajs.shape)

        # NOTE Process features through the latent world model
        ego_feat = ego_feat.reshape(batch_size * num_traj, 1, -1)
        flatten_bev_feature_multi_trajs = flatten_bev_feature_multi_trajs.reshape(
            batch_size * num_traj, self.num_plan_queries, -1
        )

        # NOTE Multiple iterations
        num_iterations = self.num_fut_timestep
        interval = 8 // num_iterations

        fut_ego_feat = ego_feat
        ego_feat_list = [fut_ego_feat]

        fut_flatten_bev_feature_multi_trajs = flatten_bev_feature_multi_trajs
        bev_feat_list = [fut_flatten_bev_feature_multi_trajs]

        for i in range(num_iterations):
            fut_ego_feat, fut_flatten_bev_feature_multi_trajs = self._latent_world_model_processing(
                fut_flatten_bev_feature_multi_trajs, fut_ego_feat, batch_size, num_traj
            )
            fut_flatten_bev_feature_multi_trajs = self._inject_fut_ego_into_bev(
                agent_trajectory,
                fut_flatten_bev_feature_multi_trajs,
                fut_ego_feat,
                num_traj,
                fut_idx=(i + 1) * interval,
            )
            ego_feat_list.append(fut_ego_feat)
            bev_feat_list.append(fut_flatten_bev_feature_multi_trajs)

        fut_ego_feat = ego_feat_list[-1]
        fut_flatten_bev_feature_multi_trajs = bev_feat_list[-1]

        # Compute reward features
        # [batch_size, num_traj, C]
        # NOTE with latent world model
        reward_feature = self._compute_reward_feature(
            ego_feat_list,
            bev_feat_list,
            batch_size,
            num_traj,
        )
        h = self.reward_trunk(reward_feature)
        sim_rewards = [torch.sigmoid(layer(h)) for layer in self.reward_out]

        sim_rewards = torch.cat(sim_rewards, dim=-1).permute(0, 2, 1)
        sim_logits = self.reward_logits(h).permute(0, 2, 1)
         
        output = {}
        output["sim_rewards"] = sim_rewards
        output["sim_logits"] = sim_logits

        return output


class RewardConvNet(nn.Module):
    def __init__(self, input_channels: int = 512, conv1_out_channels: int = 256, conv2_out_channels: int = 256):
        """
        Initialize RewardConvNet.

        Args:
            input_channels (int): Number of channels in the input feature map. Default is 512.
            conv1_out_channels (int): Number of output channels for the first convolution. Default is 256.
            conv2_out_channels (int): Number of output channels for the second convolution. Default is 256.
        """
        super(RewardConvNet, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=conv1_out_channels, 
            kernel_size=3, 
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)  # Batch normalization for the first layer
        self.relu1 = nn.ReLU()
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out_channels, 
            out_channels=conv2_out_channels, 
            kernel_size=3, 
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(conv2_out_channels)  # Batch normalization for the second layer
        self.relu2 = nn.ReLU()
        
        # Adaptive average pooling to reduce spatial dimensions to 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            x (torch.Tensor): Input feature map, shape [batch_size*num_traj, 512, 4, 8]

        Returns:
            torch.Tensor: Scoring features, shape [batch_size*num_traj, 128, 1, 1]
        """
        x = self.conv1(x)         # [batch_size*num_traj, 256, 4, 8]
        x = self.bn1(x)           # Batch normalization
        x = self.relu1(x)         # ReLU activation
        x = self.conv2(x)         # [batch_size*num_traj, 128, 4, 8]
        x = self.bn2(x)           # Batch normalization
        x = self.relu2(x)         # ReLU activation
        x = self.pool(x)          # [batch_size*num_traj, 128, 1, 1]
        return x
