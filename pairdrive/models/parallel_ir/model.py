from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from pairdrive.models.parallel_ir.backbone import TransfuserBackbone
from pairdrive.models.parallel_ir.config import ParallelIRConfig as TransfuserConfig
from navsim.common.enums import StateSE2Index
intention = 'tree'
mode = "other"


class BoundingBox2DIndex:
    POINT = slice(0, 2)
    HEADING = 2

    @staticmethod
    def size() -> int:
        return 5

class ParallelIRModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, trajectory_sampling: TrajectorySampling, config: TransfuserConfig):
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

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self.branch_tree = BranchTreeModule(
            d_model=config.tf_d_model,       
            d_ffn=config.tf_d_ffn,            
            branch_K=4,         
            horizon_T=trajectory_sampling.num_poses,  
            offset_range=config.offset_range,       
            heading_range=config.heading_range,  
            stride=2 
            )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        # trajectory = self._trajectory_head(trajectory_query)
        # output.update(trajectory)

        # agents = self._agent_head(agents_query)
        # output.update(agents)

        il_traj = features["expert_feature"]  # (B, T, 3)

        bt_out = self.branch_tree(query_out, il_traj)
        output.update(bt_out)

        return output


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {"trajectory": poses}

class LeftOffsetHead(nn.Module):
    def __init__(self, d_model, d_ffn, offset_range, heading_range):
        super().__init__()
        self.offset_range = float(offset_range)
        self.heading_range = float(heading_range)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, 2))
    def forward(self, h):
        s_raw, dth_raw = self.mlp(h).unbind(dim=1)
        s   = F.sigmoid(s_raw)                 * self.offset_range      # s >= 0
        if intention == 'random':
            dth =  torch.tanh(dth_raw)              * self.heading_range
        else:
            dth =  torch.sigmoid(dth_raw)              * self.heading_range
            dth =  dth.abs()                        # 左：Δψ>=0
        return torch.stack([s, dth], dim=-1)    # (N,2)
    
class CenterOffsetHead(nn.Module):
    def __init__(self, d_model, d_ffn, offset_range, heading_range):
        super().__init__()
        self.offset_range = float(offset_range)
        self.heading_range = float(heading_range)
        if intention == "random":
            self.mlp = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, 2))
        else:
            self.mlp = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, 1))
    def forward(self, h):
        # s_raw = self.mlp(h).unbind(dim=1)
        # s   = F.sigmoid(s_raw)                 * self.offset_range
        if intention == "random":
            s_raw, dth_raw = self.mlp(h).unbind(dim=1)
            s   = F.sigmoid(s_raw)                 * self.offset_range
            dth =  torch.tanh(dth_raw)              * self.heading_range
        else:
            s_raw = self.mlp(h).squeeze(-1)
            s   = F.sigmoid(s_raw)                 * self.offset_range
            dth = torch.zeros_like(s)               # 中：Δψ=0
        return torch.stack([s, dth], dim=-1)    # (N,2)

class BehindOffsetHead(nn.Module):
    def __init__(self, d_model, d_ffn, offset_range, heading_range):
        super().__init__()
        self.offset_range = float(offset_range)
        self.heading_range = float(heading_range)
        if intention == "random":
            self.mlp = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, 2))
        else:
            self.mlp = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, 1))
    def forward(self, h):
        # s_raw = self.mlp(h).unbind(dim=1)
        # s   = - F.sigmoid(s_raw)                 * self.offset_range
        if intention == "random":
            s_raw, dth_raw = self.mlp(h).unbind(dim=1)
            s   = - F.sigmoid(s_raw)                 * self.offset_range
            dth =  torch.tanh(dth_raw)              * self.heading_range
        else:
            s_raw = self.mlp(h).squeeze(-1)
            s   = - F.sigmoid(s_raw)                 * self.offset_range
            dth = torch.zeros_like(s)               # 中：Δψ=0
        return torch.stack([s, dth], dim=-1)    # (N,2)
    
class RightOffsetHead(nn.Module):
    def __init__(self, d_model, d_ffn, offset_range, heading_range):
        super().__init__()
        self.offset_range = float(offset_range)
        self.heading_range = float(heading_range)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, 2))
    def forward(self, h):
        s_raw, dth_raw = self.mlp(h).unbind(dim=1)
        s   = F.sigmoid(s_raw)                 * self.offset_range
        if intention == 'random':
            dth =  torch.tanh(dth_raw)              * self.heading_range
        else:
            dth =  torch.sigmoid(dth_raw)              * self.heading_range
            dth =  -dth.abs()                        # 左：Δψ>=0
        return torch.stack([s, dth], dim=-1)    # (N,2)

class BranchTreeModule(nn.Module):
    """
      key: 分支路径码(如 "0120"...)
      val: (B, T, 3) 对应此路径在所有 batch 的轨迹
    """
    def __init__(self, d_model: int, d_ffn: int, branch_K: int, horizon_T: int,
                 offset_range: float, heading_range: float, stride: int = 2):
        super().__init__()
        assert branch_K >= 3 and stride >= 1
        self.K = branch_K
        self.T = horizon_T
        self.stride = stride
        self.smooth_enabled = True
        self.smooth_win = 1          # 窗口半径 w
        self.smooth_sigma = 1.2      # 高斯核sigma通常 1.0~2.0
        self.smooth_alpha = 0.3      # 替换强度：1=全替换窗口内点；0.5=与原轨迹折中
        self.offset_range = float(offset_range)
        self.heading_range = float(heading_range)
        self.num_levels = 4
        self.anchor_encoder = nn.Sequential(
            nn.Linear(3,128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        self.branch_type_embed = nn.Embedding(branch_K, d_model)
        with torch.no_grad():
            base = torch.zeros(branch_K, d_model)
            base[0, :] = 0.0  # Left
            base[1, :] = 1.0   # Center
            base[2, :] = 2.0   # Right
            self.branch_type_embed.weight.copy_(base)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_ffn,
            dropout=0.1, batch_first=True,
        )
        self.shared_attn = nn.TransformerDecoder(dec_layer, num_layers=5)

        self.level_heads_left   = nn.ModuleList([
            LeftOffsetHead(d_model, d_ffn, offset_range, heading_range)
            for _ in range(self.num_levels)
        ])
        self.level_heads_center = nn.ModuleList([
            CenterOffsetHead(d_model, d_ffn, offset_range, heading_range)
            for _ in range(self.num_levels)
        ])
        self.level_heads_behind = nn.ModuleList([
            BehindOffsetHead(d_model, d_ffn, offset_range, heading_range)
            for _ in range(self.num_levels)
        ])
        self.level_heads_right  = nn.ModuleList([
            RightOffsetHead(d_model, d_ffn, offset_range, heading_range)
            for _ in range(self.num_levels)
        ])

        # self.head_left   = LeftOffsetHead(d_model, d_ffn, self.offset_range, self.heading_range)
        # self.head_center = CenterOffsetHead(d_model, d_ffn, self.offset_range, self.heading_range)
        # self.head_right  = RightOffsetHead(d_model, d_ffn, self.offset_range, self.heading_range)
        
        self.score_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    @torch.no_grad()
    def _repeat_context(self, ctx: torch.Tensor, times: int) -> torch.Tensor:
        return ctx if times == 1 else ctx.repeat_interleave(times, dim=0)
    
    def _gaussian_kernel1d(self, k: int, sigma: float, device, dtype):
        half = (k - 1) // 2
        x = torch.arange(-half, half + 1, device=device, dtype=dtype)
        w = torch.exp(-(x ** 2) / (2 * sigma * sigma))
        w = (w / w.sum()).view(1, 1, k)   # (1,1,k)
        return w

    def _conv1d_same(self, x1d: torch.Tensor, ker: torch.Tensor, pad_mode: str = "replicate") -> torch.Tensor:
        """
        x1d: (B*, 1, T), ker: (1,1,k)
        使用复制/镜像填充，避免边界被零拉拽导致末端“折回”。
        """
        pad = (ker.shape[-1] - 1) // 2
        if pad <= 0:
            return torch.conv1d(x1d, ker, padding=0)
        # 先在时间维做对称填充，再无 padding 卷积
        xpad = F.pad(x1d, (pad, pad), mode=pad_mode)  # "replicate" 或 "reflect"
        y = torch.conv1d(xpad, ker, padding=0)
        # 形状仍是 (B*,1,T)
        return y

    def _local_smooth_window(self, trajs: torch.Tensor, t_idx: int) -> torch.Tensor:
        """
        trajs: (B, N*, T, 3) 刚拼好的 full（包含 prefix + new_pt + tail）
        仅对 [t_idx - w, ..., t_idx + w] 时刻做平滑；其它时刻不变
        """
        if (not self.smooth_enabled) or (self.smooth_win <= 0):
            return trajs

        B, Nk, T, _ = trajs.shape
        dev, dt = trajs.device, trajs.dtype
        w = self.smooth_win
        L = max(0, t_idx - w)
        R = min(T - 1, t_idx + w)
        if L >= R:
            return trajs

        ksize = 2 * w + 1
        ker = self._gaussian_kernel1d(ksize, self.smooth_sigma, dev, dt)  # (1,1,k)

        # === 平滑出一份“候选轨迹” ===
        # x/y
        xy = trajs[..., :2].reshape(B * Nk, T, 2)
        x = xy[..., 0].unsqueeze(1)  # (BN,1,T)
        y = xy[..., 1].unsqueeze(1)
        x_s = self._conv1d_same(x, ker).squeeze(1)  # (BN,T)
        y_s = self._conv1d_same(y, ker).squeeze(1)

        # heading：先 sin/cos 再还原，避免跨 π 折返
        th = trajs[..., 2].reshape(B * Nk, T)              # (BN,T)
        c  = self._conv1d_same(torch.cos(th).unsqueeze(1), ker).squeeze(1)
        s  = self._conv1d_same(torch.sin(th).unsqueeze(1), ker).squeeze(1)
        th_s = torch.atan2(s, c)                           # (BN,T)

        traj_s = torch.stack([x_s, y_s, th_s], dim=-1).reshape(B, Nk, T, 3)

        # === 只在窗口内替换/融合 ===
        alpha = float(self.smooth_alpha)
        mask = torch.zeros((1, 1, T, 1), device=dev, dtype=dt)
        mask[:, :, L:R + 1, :] = alpha
        trajs = trajs * (1.0 - mask) + traj_s * mask
        return trajs

    def forward(self, context_feat: torch.Tensor, expert_traj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        context_feat : (B, N_ctx, d_model)
        expert_traj  : (B, T, 3)
        return       : dict[str -> (B, T, 3)]
        """
        # dtype/device 对齐
        dtype  = next(self.parameters()).dtype
        device = next(self.parameters()).device
        context_feat = context_feat.to(dtype=dtype, device=device)#(64,31,128)
        expert_traj  = expert_traj.to(dtype=dtype, device=device)

        B, T, _ = expert_traj.shape #(64,8,3)
        assert T >= self.T
        
        xp = expert_traj[..., 0]; yp = expert_traj[..., 1]; thp = expert_traj[..., 2] 
        dx_L = xp[:, 1:] - xp[:, :-1]; dy_L = yp[:, 1:] - yp[:, :-1]; d_th  = thp[:, 1:] - thp[:, :-1]
        step_len = torch.sqrt(dx_L * dx_L + dy_L * dy_L)              # (B, T-1)
        avg_step = step_len.mean(dim=1)                       # (B,)
        #自车坐标系
        ct = torch.cos(thp[:, :-1])           # (B, T-1)
        st = torch.sin(thp[:, :-1])
        dx_B =  ct * dx_L + st * dy_L         # (B, T-1)
        dy_B = -st * dx_L + ct * dy_L         # (B, T-1)
        loc_xy = torch.stack([dx_B, dy_B], dim=-1)  # (B, T-1, 2)

        # 轨迹集合 & 路径码集合
        all_trajs = expert_traj[:, None, :, :]                      # (B, N=1, T, 3)
        all_codes = torch.zeros((B, 1, 0), dtype=torch.long, device=device)  # (B, N, L=0)
        logits_levels: list[torch.Tensor] = []   # (B, num_parents, K)
        level_idx = 0

        GRPO_num = 15
        stage = 1
        if stage == 1:
            if GRPO_num == 15:
                code_list = ["0000", "0001", "0011", "1001", "1100", "1111", "3333", "1133",
                            "1222","1122","2211","2221", "2222", "2201", "2210"]
            elif GRPO_num == 12:
                code_list = ["0000", "0011", "1100", "1111", "3333", "1133",
                            "1122","2211", "2222", "2221", "2111", "2210"]
            elif GRPO_num == 9:
                code_list = ["0000", "0011", "1100", "1111", "1133",
                            "1112","2211", "2222", "2210"]
            elif GRPO_num == 5:
                code_list = ["0000", "1111", "2222", "0011", "2211"]
            else:
                raise ValueError("GRPO_num must be in [15, 12, 10, 5]")
        elif stage == 2:
            if GRPO_num == 15:
                code_list = ["1111", "3333", "1133", "1113"]
            elif GRPO_num == 12:
                code_list = ["1111", "3333", "1133"]
            elif GRPO_num == 9:
                code_list = ["1111", "1133"]
            elif GRPO_num == 5:
                code_list = ["1111"]
            else:
                raise ValueError("GRPO_num must be in [15, 12, 10, 5]")
        
        # stage two
        # code_list = ["1111", "3333", "1113", "1133"]
        # 分叉循环（仅在 t % stride==0 时扩展）
        for t in range(self.T):
            if ( (t+1) % self.stride) != 0:
                continue

            _, N, _, _ = all_trajs.shape #N=1
            anchors = all_trajs[:, :, t, :].reshape(B * N, 3)       # (B*N,3)!!!!!!!!!

            anchor_emb = self.anchor_encoder(anchors)               # (B*N,d)(B,128)
            branch_ids = torch.arange(self.K, device=device)[None, :].repeat(B * N, 1)  # (B*N,K)(B,3)
            branch_emb = self.branch_type_embed(branch_ids)         # (B*N,K,d)(B,3,128)
            queries = anchor_emb[:, None, :] + branch_emb           # (B*N,K,d)(B,3,128)
            ctx_rep = self._repeat_context(context_feat, times=N)   # (B*N,N_ctx,d)
            h = self.shared_attn(queries, ctx_rep)                  # (B*N,K,d)(B,3,128)

            theta_base = all_trajs[:, :, t, 2].reshape(B * N)   # (B*N,)
            theta_exp  = thp[:, t].repeat_interleave(N)
            logits_this = []
            for k in range(self.K):
                hk = h[:, k, :]                         # (B*N, d)
                logits_this.append(self.score_head(hk)) # (B*N, 1)
            logits_this = torch.cat(logits_this, dim=1) # (B*N, K)
            logits_levels.append(logits_this.view(B, N, self.K))

            self.head_left = self.level_heads_left[level_idx]
            self.head_center = self.level_heads_center[level_idx]
            self.head_right = self.level_heads_right[level_idx]
            self.head_behind = self.level_heads_behind[level_idx]

            # 逐分支偏移
            deltas_list = []
            for k in range(self.K):
                hk = h[:, k, :]
                if k == 0:   sd = self.head_left(hk)
                elif k == 1: sd = self.head_center(hk)
                elif k == 2: sd = self.head_right(hk)
                elif k == 3: sd = self.head_behind(hk)
                else:
                    raise NotImplementedError(f"Unsupported branch id {k}")
                s   = sd[:, 0]
                dth = sd[:, 1]
                # 按照速度缩放
                scale_rep = (avg_step * 1.5).repeat_interleave(N).to(s.device, s.dtype)  # (B*N,)
                s = s * scale_rep

                base =  theta_base
                theta_k = base + dth
                dx  = s * torch.cos(theta_k)
                dy  = s * torch.sin(theta_k)
                delta = torch.stack([dx, dy, dth], dim=-1)   # Δx, Δy, Δθ
                deltas_list.append(delta)
            deltas = torch.stack(deltas_list, dim=1).reshape(B, N, self.K, 3)  # (B,N,K,3)
            # 新点
            new_pts = all_trajs[:, :, t, :][:, :, None, :] + deltas            # (B,N,K,3)

            # 嫁接尾巴 & 组合
            if t < self.T - 1:
                rem_rel  = expert_traj[:, t+1:, :] - expert_traj[:, t:t+1, :]   
                prefix   = all_trajs[:, :, :t, :]                                  # (B, N, t, 3)
                prefix_k = prefix[:, :, None, :, :].expand(-1, -1, self.K, -1, -1) # (B, N, K, t, 3)
                new_pt_b = new_pts[:, :, :, None, :]                               # (B, N, K, 1, 3)
                tail     = rem_rel[:, None, None, :, :] + new_pts[:, :, :, None, :]# (B, N, K, T-t-1, 3)
                full     = torch.cat([prefix_k, new_pt_b, tail], dim=3).reshape(B, N*self.K, self.T, 3)

            else:
                prefix   = all_trajs[:, :, :t, :]                              # (B, N, T-1, 3)
                prefix_k = prefix[:, :, None, :, :].expand(-1, -1, self.K, -1, -1)    # (B,N,K,T-1,3)
                new_pt_b = new_pts[:, :, :, None, :]                           # (B,N,K,1,3)
                full     = torch.cat([prefix_k, new_pt_b], dim=3).reshape(B, N * self.K, self.T, 3)
            
            # full = self._local_smooth_window(full, t)
            # 更新路径码（在末尾追加当前分支编号）
            code_len = all_codes.shape[-1]
            codes_expanded = all_codes[:, :, None, :].expand(B, N, self.K, code_len)  # (B,N,K,L)
            branch_choices = torch.arange(self.K, device=device).view(1, 1, self.K, 1).expand(B, N, self.K, 1)
            new_codes = torch.cat([codes_expanded, branch_choices], dim=-1).reshape(B, N * self.K, code_len + 1)

            all_trajs = full
            all_codes = new_codes
            level_idx += 1

        # === 组装输出字典：key=路径码，value=(B,T,3) ===
        result: Dict[str, torch.Tensor] = {}
        N_final = all_trajs.shape[1]
        # 假设各 batch 的 code 顺序一致（展开顺序相同）；用第0个batch的 code 作为 key
        codes0 = all_codes[0]                        # (N_final, L)
        for i in range(N_final):
            key = "".join(str(int(x)) for x in codes0[i].tolist())
            if mode == "inference":
                result[key] = all_trajs[:, i, :, :]
            else:
                if (code_list is None) or (key in code_list):
                    result[key] = all_trajs[:, i, :, :]  # (B, T, 3)

        return {"trajectory":result, "logits_levels": logits_levels}
