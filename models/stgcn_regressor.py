"""
Minimal ST-GCN regressor for CoP prediction.

Uses skeleton graph-temporal processing on pose streams, with optional VR fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _normalize_adjacency(a: torch.Tensor) -> torch.Tensor:
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = a.sum(dim=1).clamp_min(1e-6)
    d_inv_sqrt = torch.pow(deg, -0.5)
    d_mat = torch.diag(d_inv_sqrt)
    return d_mat @ a @ d_mat


def _default_adjacency(n_joints: int) -> torch.Tensor:
    # Chain + self loops (works for arbitrary ordered joints)
    a = torch.eye(n_joints, dtype=torch.float32)
    for i in range(n_joints - 1):
        a[i, i + 1] = 1.0
        a[i + 1, i] = 1.0
    return _normalize_adjacency(a)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # x: (B, C, W, J), a: (J, J)
        x = torch.einsum("bcwj,jk->bcwk", x, a)
        return self.temporal(x)


class STGCNCoPPredictor(nn.Module):
    def __init__(
        self,
        n_pose_joints: int = 8,
        n_vel_joints: int = 8,
        n_vr_coords: int = 9,
        use_vr_vel: bool = True,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_pose_joints = int(n_pose_joints)
        self.n_vel_joints = int(n_vel_joints)
        self.n_vr_coords = int(n_vr_coords)
        self.use_pose_vel = self.n_vel_joints > 0
        self.use_vr_vel = bool(use_vr_vel) and self.n_vr_coords > 0

        in_ch = 3 + (3 if self.use_pose_vel else 0)
        mid_ch = max(hidden_size // 2, 32)

        self.register_buffer("adjacency", _default_adjacency(self.n_pose_joints))
        self.block1 = STGCNBlock(in_ch, mid_ch, dropout)
        self.block2 = STGCNBlock(mid_ch, hidden_size, dropout)

        vr_in = self.n_vr_coords + (self.n_vr_coords if self.use_vr_vel else 0)
        if vr_in > 0:
            self.vr_encoder = nn.GRU(
                input_size=vr_in,
                hidden_size=max(hidden_size // 2, 32),
                num_layers=max(num_layers, 1),
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            fusion_in = hidden_size + max(hidden_size // 2, 32)
        else:
            self.vr_encoder = None
            fusion_in = hidden_size

        self.head = nn.Sequential(
            nn.Linear(fusion_in, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, pose_pos, pose_vel, vr_pos, vr_vel):
        # pose_pos: (B, W, J, 3)
        b, w, j, _ = pose_pos.shape
        if j != self.n_pose_joints:
            raise ValueError(f"pose_pos joints mismatch: expected {self.n_pose_joints}, got {j}")

        if self.use_pose_vel:
            if pose_vel is None:
                raise ValueError("Model expects pose_vel but got None")
            x = torch.cat([pose_pos, pose_vel], dim=-1)  # (B, W, J, C)
        else:
            x = pose_pos

        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, W, J)
        x = self.block1(x, self.adjacency)
        x = self.block2(x, self.adjacency)
        x = x.mean(dim=(2, 3))  # (B, H)

        if self.vr_encoder is not None:
            if vr_pos is None:
                raise ValueError("Model expects vr_pos but got None")
            vr_parts = [vr_pos]
            if self.use_vr_vel:
                if vr_vel is None:
                    raise ValueError("Model expects vr_vel but got None")
                vr_parts.append(vr_vel)
            v = torch.cat(vr_parts, dim=-1)  # (B, W, Cv)
            v, _ = self.vr_encoder(v)
            v = v[:, -1, :]
            x = torch.cat([x, v], dim=-1)

        return self.head(x)

