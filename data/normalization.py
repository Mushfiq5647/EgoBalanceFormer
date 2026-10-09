"""
Input / output normalization for balance prediction.

Default behavior:
  - Inputs: feature-wise z-score normalization.
    Example:
      pose_pos (W, J, 3) -> mean/std shape (J, 3), reduced across dataset+time.
      vr_pos   (W, C)    -> mean/std shape (C,), reduced across dataset+time.
  - Target (cop): per-axis normalization (x,y), configurable:
      * zscore: (cop - mean_xy) / std_xy
      * minmax: maps each axis to [-1, 1] using train-set min/max.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch


class Normalizer:
    """
    Z-score normalization for each input modality and the CoP target.

    Fit on training data once, then apply identically to val/test.
    """

    def __init__(self, cop_mode: str = "zscore") -> None:
        self._stats: Dict[str, Dict[str, np.ndarray]] = {}
        self.cop_mode = cop_mode

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, dataset) -> "Normalizer":
        """
        Iterate over the full dataset (no DataLoader needed) and accumulate
        mean / std for each key in the sample dict.

        Args:
            dataset: BalanceDataset instance (or any map-style dataset whose
                     __getitem__ returns dicts of tensors)
        """
        print(f"[Normalizer] Fitting on {len(dataset)} samples …")
        accum: Dict[str, list] = {}

        # Keys to skip (not model inputs/outputs)
        SKIP_KEYS = {"cop_mean"}

        for i in range(len(dataset)):
            sample = dataset[i]
            for key, val in sample.items():
                if key in SKIP_KEYS:
                    continue
                if not isinstance(val, torch.Tensor):
                    continue
                accum.setdefault(key, []).append(val.numpy().astype(np.float64))

        for key, arrays in accum.items():
            stacked = np.stack(arrays, axis=0)  # (N, ...)
            if key == "cop":
                stacked = np.stack(arrays, axis=0)  # (N, 2)
                if self.cop_mode == "minmax":
                    vmin = stacked.min(axis=0).astype(np.float32)
                    vmax = stacked.max(axis=0).astype(np.float32)
                    scale = np.clip(vmax - vmin, 1e-6, None).astype(np.float32)
                    self._stats[key] = {"min": vmin, "max": vmax, "scale": scale}
                    print(f"  {key:12s}: min={vmin.tolist()}  max={vmax.tolist()}  mode=minmax[-1,1]")
                else:
                    mean = stacked.mean(axis=0).astype(np.float32)
                    std = np.clip(stacked.std(axis=0), 1e-6, None).astype(np.float32)
                    self._stats[key] = {"mean": mean, "std": std}
                    print(f"  {key:12s}: mean={mean.tolist()}  std={std.tolist()}  mode=zscore")
            else:
                # Feature-wise normalization:
                # reduce across sample axis and temporal axis only.
                # sample shapes are typically (W, ...). stacked is (N, W, ...).
                # We keep trailing feature dimensions (e.g. (J,3), (C,), (3,), ...).
                if stacked.ndim >= 2:
                    reduce_axes = (0, 1)
                else:
                    reduce_axes = (0,)
                mean = stacked.mean(axis=reduce_axes).astype(np.float32)
                std = np.clip(stacked.std(axis=reduce_axes), 1e-6, None).astype(np.float32)
                self._stats[key] = {"mean": mean, "std": std}
                print(
                    f"  {key:12s}: mean_shape={tuple(mean.shape)}  "
                    f"std_shape={tuple(std.shape)}  mode=feature_zscore"
                )

        return self

    # ── Apply ─────────────────────────────────────────────────────────────────

    def normalize_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize all tensor keys in a batch dict in-place (returns same dict).
        Note: 'cop_mean' is intentionally excluded — it's a reconstruction helper."""
        SKIP_KEYS = {"cop_mean"}
        for key, val in batch.items():
            if key in SKIP_KEYS:
                continue
            if not isinstance(val, torch.Tensor):
                continue
            if key not in self._stats:
                continue
            if key == "cop" and self.cop_mode == "minmax":
                min_np = self._stats[key]["min"]
                scale_np = self._stats[key]["scale"]
                min_v = torch.as_tensor(min_np, device=val.device, dtype=val.dtype)
                scale = torch.as_tensor(scale_np, device=val.device, dtype=val.dtype)
                batch[key] = 2.0 * ((val - min_v) / scale) - 1.0
            else:
                mean_np = self._stats[key]["mean"]
                std_np  = self._stats[key]["std"]
                mean = torch.as_tensor(mean_np, device=val.device, dtype=val.dtype)
                std  = torch.as_tensor(std_np, device=val.device, dtype=val.dtype)
                batch[key] = (val - mean) / std
        return batch

    def denormalize_cop(self, cop_normalized: torch.Tensor) -> torch.Tensor:
        """Inverse-transform normalized CoP predictions back to original units."""
        if "cop" not in self._stats:
            return cop_normalized
        if self.cop_mode == "minmax":
            min_v = torch.as_tensor(
                self._stats["cop"]["min"],
                device=cop_normalized.device,
                dtype=cop_normalized.dtype,
            )
            scale = torch.as_tensor(
                self._stats["cop"]["scale"],
                device=cop_normalized.device,
                dtype=cop_normalized.dtype,
            )
            return ((cop_normalized + 1.0) * 0.5) * scale + min_v
        mean = torch.as_tensor(
            self._stats["cop"]["mean"],
            device=cop_normalized.device,
            dtype=cop_normalized.dtype,
        )
        std = torch.as_tensor(
            self._stats["cop"]["std"],
            device=cop_normalized.device,
            dtype=cop_normalized.dtype,
        )
        return cop_normalized * std + mean

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **{
            f"{key}__{stat}": val
            for key, stats in self._stats.items()
            for stat, val in stats.items()
        }, __cop_mode=np.array(self.cop_mode))
        print(f"[Normalizer] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        norm = cls()
        data = np.load(str(path))
        if "__cop_mode" in data.files:
            norm.cop_mode = str(data["__cop_mode"].item())
        for full_key in data.files:
            if full_key == "__cop_mode":
                continue
            key, stat = full_key.rsplit("__", 1)
            norm._stats.setdefault(key, {})[stat] = data[full_key]
        print(f"[Normalizer] Loaded from {path}")
        return norm

    def is_fitted(self) -> bool:
        return len(self._stats) > 0
