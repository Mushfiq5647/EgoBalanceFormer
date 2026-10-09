"""
Simple GRU time-series regressor for CoP prediction.

Input per frame is the concatenation of available modalities:
  - pose_pos (required)
  - pose_vel (optional)
  - vr_pos   (required)
  - vr_vel   (optional)
"""

import torch
import torch.nn as nn


class GRUCoPPredictor(nn.Module):
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
        self.n_pose_joints = n_pose_joints
        self.n_vel_joints = n_vel_joints
        self.n_vr_coords = n_vr_coords
        self.use_pose_vel = n_vel_joints > 0
        self.use_vr_vel = use_vr_vel

        pos_dim = n_pose_joints * 3
        vel_dim = n_vel_joints * 3 if self.use_pose_vel else 0
        vr_dim = n_vr_coords
        vv_dim = n_vr_coords if self.use_vr_vel else 0
        self.input_dim = pos_dim + vel_dim + vr_dim + vv_dim

        self.gru = nn.GRU(
            input_size=self.input_dim,
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
        B, W = pose_pos.shape[:2]
        if pose_pos.shape[2] != self.n_pose_joints:
            raise ValueError(
                f"pose_pos joints mismatch: expected {self.n_pose_joints}, got {pose_pos.shape[2]}"
            )

        parts = [pose_pos.reshape(B, W, -1)]
        if self.use_pose_vel:
            if pose_vel is None:
                raise ValueError("Model expects pose_vel but got None")
            if pose_vel.shape[2] != self.n_vel_joints:
                raise ValueError(
                    f"pose_vel joints mismatch: expected {self.n_vel_joints}, got {pose_vel.shape[2]}"
                )
            parts.append(pose_vel.reshape(B, W, -1))
        if self.n_vr_coords > 0:
            if vr_pos is None:
                raise ValueError("Model expects vr_pos but got None")
            parts.append(vr_pos)
        if self.use_vr_vel and self.n_vr_coords > 0:
            if vr_vel is None:
                raise ValueError("Model expects vr_vel but got None")
            parts.append(vr_vel)

        x = torch.cat(parts, dim=-1)  # (B, W, F)
        y, _ = self.gru(x)            # (B, W, H)
        return self.head(y[:, -1, :])  # (B, 2)
