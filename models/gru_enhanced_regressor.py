"""
Enhanced GRU baseline using the same feature set as st_transformer_enhanced.

This model flattens all enabled per-frame modalities and feeds the resulting
sequence to a GRU. It is intended as a matched-input temporal baseline for the
enhanced spatio-temporal transformer.
"""

import torch
import torch.nn as nn


class GRUEnhancedCoPPredictor(nn.Module):
    def __init__(
        self,
        n_joints: int = 8,
        n_groups: int = 4,
        n_vr_coords: int = 0,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_activation: str = "linear",
        use_joint_groups: bool = True,
        use_joint_rotations: bool = True,
        use_rotation_vel: bool = True,
        use_pose_vel: bool = True,
        use_com: bool = True,
    ):
        super().__init__()
        self.n_joints = int(n_joints)
        self.n_groups = int(n_groups)
        self.n_vr_coords = int(n_vr_coords)
        self.use_joint_groups = bool(use_joint_groups)
        self.use_joint_rotations = bool(use_joint_rotations)
        self.use_rotation_vel = bool(use_rotation_vel)
        self.use_pose_vel = bool(use_pose_vel)
        self.use_com = bool(use_com)
        self.use_vr = self.n_vr_coords > 0
        self.output_activation = output_activation

        input_dim = self.n_joints * 3
        if self.use_pose_vel:
            input_dim += self.n_joints * 3
        if self.use_vr:
            input_dim += self.n_vr_coords * 2
        if self.use_joint_groups:
            input_dim += self.n_groups * 3
        if self.use_joint_rotations:
            input_dim += self.n_joints * 3
            if self.use_rotation_vel:
                input_dim += self.n_joints * 3
        if self.use_com:
            input_dim += 6

        self.input_dim = input_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def _flat(self, x: torch.Tensor, name: str, b: int, w: int) -> torch.Tensor:
        if x is None:
            raise ValueError(f"Enhanced GRU expected {name} but got None")
        if x.shape[:2] != (b, w):
            raise ValueError(
                f"{name} time shape mismatch: expected {(b, w)}, got {tuple(x.shape[:2])}"
            )
        return x.reshape(b, w, -1)

    def forward(
        self,
        pose_pos: torch.Tensor,
        pose_vel: torch.Tensor | None,
        vr_pos: torch.Tensor | None,
        vr_vel: torch.Tensor | None,
        joint_groups: torch.Tensor | None = None,
        joint_rotations: torch.Tensor | None = None,
        joint_rot_vel: torch.Tensor | None = None,
        com: torch.Tensor | None = None,
        com_vel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, w, j, _ = pose_pos.shape
        if j != self.n_joints:
            raise ValueError(
                f"pose_pos joints mismatch: expected {self.n_joints}, got {j}"
            )

        parts = [self._flat(pose_pos, "pose_pos", b, w)]
        if self.use_pose_vel:
            parts.append(self._flat(pose_vel, "pose_vel", b, w))
        if self.use_vr:
            parts.append(self._flat(vr_pos, "vr_pos", b, w))
            parts.append(self._flat(vr_vel, "vr_vel", b, w))
        if self.use_joint_groups:
            if joint_groups is None or joint_groups.shape[2] != self.n_groups:
                got = None if joint_groups is None else joint_groups.shape[2]
                raise ValueError(
                    f"joint_groups mismatch: expected {self.n_groups}, got {got}"
                )
            parts.append(self._flat(joint_groups, "joint_groups", b, w))
        if self.use_joint_rotations:
            parts.append(self._flat(joint_rotations, "joint_rotations", b, w))
            if self.use_rotation_vel:
                parts.append(self._flat(joint_rot_vel, "joint_rot_vel", b, w))
        if self.use_com:
            parts.append(self._flat(com, "com", b, w))
            if com_vel is None:
                com_vel = torch.zeros_like(com)
            parts.append(self._flat(com_vel, "com_vel", b, w))

        x = torch.cat(parts, dim=-1)
        y, _ = self.gru(x)
        pred = self.head(y[:, -1, :])
        if self.output_activation == "tanh":
            pred = torch.tanh(pred)
        return pred
