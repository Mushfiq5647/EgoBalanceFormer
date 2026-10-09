"""
Minimal CNN-LSTM regressor for CoP prediction.

Pipeline:
  1) Concatenate per-frame inputs into (B, W, F)
  2) Temporal 1D convolution on features -> (B, C, W)
  3) LSTM over time -> (B, W, H)
  4) Regress from last timestep -> (B, 2)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNLSTMCoPPredictor(nn.Module):
    def __init__(
        self,
        n_pose_joints: int = 8,
        n_vel_joints: int = 8,
        n_vr_coords: int = 9,
        use_vr_vel: bool = True,
        conv_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_pose_joints = int(n_pose_joints)
        self.n_vel_joints = int(n_vel_joints)
        self.n_vr_coords = int(n_vr_coords)
        self.use_pose_vel = self.n_vel_joints > 0
        self.use_vr_vel = bool(use_vr_vel)

        pos_dim = self.n_pose_joints * 3
        vel_dim = self.n_vel_joints * 3 if self.use_pose_vel else 0
        vr_dim = self.n_vr_coords
        vv_dim = self.n_vr_coords if (self.use_vr_vel and self.n_vr_coords > 0) else 0
        self.input_dim = pos_dim + vel_dim + vr_dim + vv_dim

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.input_dim, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
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
            if self.use_vr_vel:
                if vr_vel is None:
                    raise ValueError("Model expects vr_vel but got None")
                parts.append(vr_vel)

        x = torch.cat(parts, dim=-1)   # (B, W, F)
        x = x.transpose(1, 2)          # (B, F, W)
        x = self.temporal_conv(x)      # (B, C, W)
        x = x.transpose(1, 2)          # (B, W, C)
        y, _ = self.lstm(x)            # (B, W, H)
        return self.head(y[:, -1, :])  # (B, 2)

