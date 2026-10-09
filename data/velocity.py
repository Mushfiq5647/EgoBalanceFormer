"""
Velocity computation from position sequences using finite differences.
"""

import numpy as np


class VelocityComputer:
    """
    Computes velocities from (T, ...) position arrays using finite differences.

    Handles any shape where T is the FIRST axis:
        (T, N, 3)  - joints or VR devices
        (T, D)     - flat feature vectors

    Example with fps=22:
        dt = 1/22 ≈ 0.0454 s
        Central difference at frame t:
            v[t] = (pos[t+1] - pos[t-1]) / (2 * dt)
    """

    def __init__(self, fps: float = 22.0):
        self.fps = fps
        self.dt = 1.0 / fps

    def compute(
        self,
        positions: np.ndarray,
        method: str = "central",
        smooth_window: int = 3,
    ) -> np.ndarray:
        """
        Args:
            positions:     (T, ...) array, T is time axis
            method:        'central' | 'forward' | 'backward'
            smooth_window: moving-average kernel size (1 = no smoothing)

        Returns:
            velocities:    same shape as positions, units = input_units / second
        """
        T = positions.shape[0]
        vel = np.zeros_like(positions, dtype=np.float32)

        if method == "central":
            vel[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * self.dt)
            vel[0] = (positions[1] - positions[0]) / self.dt
            vel[-1] = (positions[-1] - positions[-2]) / self.dt
        elif method == "forward":
            vel[:-1] = (positions[1:] - positions[:-1]) / self.dt
            vel[-1] = vel[-2]
        elif method == "backward":
            vel[1:] = (positions[1:] - positions[:-1]) / self.dt
            vel[0] = vel[1]
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'central', 'forward', or 'backward'.")

        if smooth_window > 1:
            vel = self._moving_average(vel, smooth_window)

        return vel

    @staticmethod
    def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
        """Apply symmetric moving-average over axis 0 (time)."""
        T = arr.shape[0]
        out = np.zeros_like(arr)
        half = window // 2
        for t in range(T):
            lo = max(0, t - half)
            hi = min(T, t + half + 1)
            out[t] = arr[lo:hi].mean(axis=0)
        return out
