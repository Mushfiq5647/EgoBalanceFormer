"""
Simple DeepTCN regressor for CoP prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Remove right padding to preserve causal length."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.downsample(x))


class DeepTCNCoPPredictor(nn.Module):
    def __init__(
        self,
        n_pose_joints: int = 8,
        n_vel_joints: int = 8,
        n_vr_coords: int = 9,
        use_vr_vel: bool = True,
        hidden_channels: int = 128,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_pose_joints = int(n_pose_joints)
        self.n_vel_joints = int(n_vel_joints)
        self.use_pose_vel = self.n_vel_joints > 0
        self.n_vr_coords = int(n_vr_coords)
        self.use_vr_vel = bool(use_vr_vel)

        pos_dim = self.n_pose_joints * 3
        vel_dim = self.n_vel_joints * 3 if self.use_pose_vel else 0
        vr_dim = self.n_vr_coords
        vv_dim = self.n_vr_coords if self.use_vr_vel else 0
        self.input_dim = pos_dim + vel_dim + vr_dim + vv_dim

        layers = []
        in_ch = self.input_dim
        for i in range(num_levels):
            dilation = 2 ** i
            out_ch = hidden_channels
            layers.append(
                TemporalBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2),
        )

    def forward(self, pose_pos, pose_vel, vr_pos, vr_vel):
        b, w = pose_pos.shape[:2]
        parts = [pose_pos.reshape(b, w, -1)]

        if self.use_pose_vel:
            if pose_vel is None:
                raise ValueError("Model expects pose_vel but got None")
            parts.append(pose_vel.reshape(b, w, -1))

        if self.n_vr_coords > 0:
            if vr_pos is None:
                raise ValueError("Model expects vr_pos but got None")
            parts.append(vr_pos)
        if self.use_vr_vel and self.n_vr_coords > 0:
            if vr_vel is None:
                raise ValueError("Model expects vr_vel but got None")
            parts.append(vr_vel)

        x = torch.cat(parts, dim=-1)    # (B, W, F)
        x = x.transpose(1, 2)           # (B, F, W)
        y = self.tcn(x)                 # (B, H, W)
        last = y[:, :, -1]              # (B, H)
        return self.head(last)          # (B, 2)
