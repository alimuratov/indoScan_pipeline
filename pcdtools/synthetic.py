from __future__ import annotations

"""Synthetic data helpers for tests.

Utilities to create clusters and inject apparent outliers below (deeper Z) the
main cluster. Designed for deterministic unit tests of filtering logic.
"""

import numpy as np
from typing import Tuple


def inject_downward_outliers(
    points: np.ndarray,
    *,
    num_outliers: int = 20,
    z_delta: float = 0.15,
    xy_jitter: float = 0.02,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject obvious outliers below the input cluster along -Z.

    Args:
        points: (N,3) inliers representing the pothole cluster (z negative).
        num_outliers: how many outliers to add.
        z_delta: how much deeper (in meters) the outliers are pushed.
        xy_jitter: jitter added in XY so outliers are not stacked exactly.
        seed: RNG seed for determinism.

    Returns:
        (combined_points, outlier_mask): combined array (N+M,3) and boolean mask
        of shape (N+M,) where True indicates an injected outlier.
    """
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    if N == 0 or num_outliers <= 0:
        return points.copy(), np.zeros((N,), dtype=bool)

    # Sample anchors from existing points
    idx = rng.integers(0, N, size=(num_outliers,))
    anchors = points[idx]

    # Jitter XY slightly, push Z downward by z_delta
    jitter = rng.normal(0.0, xy_jitter, size=(num_outliers, 2))
    out_xy = anchors[:, :2] + jitter
    out_z = anchors[:, 2] - abs(z_delta)
    outliers = np.column_stack([out_xy, out_z])

    combined = np.vstack([points, outliers])
    mask = np.zeros((combined.shape[0],), dtype=bool)
    mask[N:] = True
    return combined, mask


