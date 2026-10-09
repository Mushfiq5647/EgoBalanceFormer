"""
Frozen AutoEncoder-based pose estimator used by balance dataset loading.

This module runs heatmap -> pose inference and returns 15-joint pose tensors
without writing any intermediate JSON files.
"""

import os
import sys
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pose_estimation.model.network import AutoEncoder  # noqa: E402


def _make_opt(num_heatmap: int = 15, ae_hidden_size: int = 20, num_joints: int = 15):
    return SimpleNamespace(
        num_heatmap=num_heatmap,
        ae_hidden_size=ae_hidden_size,
        num_joints=num_joints,
    )


class AEPoseEstimator:
    """
    Wrapper around pretrained AutoEncoder.predict_pose for 15-joint pose output.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        ae_hidden_size: int = 20,
        use_body_part: bool = False,
    ):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: '{checkpoint}'")

        self.device = torch.device(device)
        self.use_body_part = use_body_part

        # Monocular AE: input_channel_scale=1 → 15ch heatmaps (+optional 4ch body part)
        extra_ch = 4 if use_body_part else 0
        opt = _make_opt(num_heatmap=15, ae_hidden_size=ae_hidden_size, num_joints=15)
        self.net = AutoEncoder(opt, input_channel_scale=1, extra_input_channels=extra_ch).to(self.device)
        state = torch.load(checkpoint, map_location="cpu")
        self.net.load_state_dict(state)
        self.net.eval()

    @torch.no_grad()
    def estimate(
        self,
        heatmaps15: np.ndarray,                 # (N, 15, 64, 64)
        body_parts4: Optional[np.ndarray] = None,  # (N, 4, 64, 64)
        batch_size: int = 128,
    ) -> np.ndarray:
        if heatmaps15.ndim != 4 or heatmaps15.shape[1:] != (15, 64, 64):
            raise ValueError(f"Expected heatmaps15 shape (N,15,64,64), got {heatmaps15.shape}")

        if self.use_body_part:
            if body_parts4 is None:
                raise ValueError("AEPoseEstimator configured with use_body_part=True but no body_parts4 provided.")
            if body_parts4.ndim != 4 or body_parts4.shape[1:] != (4, 64, 64):
                raise ValueError(f"Expected body_parts4 shape (N,4,64,64), got {body_parts4.shape}")

        outputs = []
        n = heatmaps15.shape[0]
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            hm15 = torch.from_numpy(heatmaps15[lo:hi]).to(self.device, dtype=torch.float32)  # (B,15,64,64)

            if self.use_body_part:
                bp4 = torch.from_numpy(body_parts4[lo:hi]).to(self.device, dtype=torch.float32)  # (B,4,64,64)
                ae_input = torch.cat([hm15, bp4], dim=1)  # (B,19,64,64)
            else:
                ae_input = hm15

            pred = self.net.predict_pose(ae_input)   # (B, 15, 3)
            outputs.append(pred.cpu().numpy().astype(np.float32))

        return np.concatenate(outputs, axis=0)
