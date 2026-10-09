"""
Simple linear regression baseline for CoP prediction.
"""

import torch
import torch.nn as nn


class LinearCoPPredictor(nn.Module):
    def __init__(
        self,
        n_pose_joints: int = 8,
        n_vel_joints: int = 8,
        n_vr_coords: int = 9,
        use_vr_vel: bool = True,
        window_size: int = 11,
    ):
        super().__init__()
        self.n_pose_joints = n_pose_joints
        self.n_vel_joints = n_vel_joints
        self.n_vr_coords = n_vr_coords
        self.use_pose_vel = n_vel_joints > 0
        self.use_vr_vel = use_vr_vel
        self.window_size = window_size

        pos_dim = n_pose_joints * 3
        vel_dim = n_vel_joints * 3 if self.use_pose_vel else 0
        vr_dim = n_vr_coords
        vv_dim = n_vr_coords if self.use_vr_vel else 0
        self.step_dim = pos_dim + vel_dim + vr_dim + vv_dim
        self.input_dim = self.step_dim * window_size

        self.head = nn.Linear(self.input_dim, 2)

    def forward(self, pose_pos, pose_vel, vr_pos, vr_vel):
        b, w = pose_pos.shape[:2]
        if w != self.window_size:
            raise ValueError(f"Window mismatch: expected {self.window_size}, got {w}")

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

        x = torch.cat(parts, dim=-1).reshape(b, -1)  # (B, W*F)
        return self.head(x)  # (B, 2)
