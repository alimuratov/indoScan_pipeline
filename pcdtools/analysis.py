from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict


def convex_hull_area(points_xy: np.ndarray) -> Tuple[float, Optional[object]]:
    """Return (area, hull_object) of the convex hull in 2D.

    For 2D, scipy ConvexHull.volume equals polygon area.
    """
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points_xy)
        return float(hull.volume), hull
    except Exception:
        return 0.0, None


def fit_quadratic_surface(points: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Least-squares fit of z = ax^2 + by^2 + cxy + dx + ey + f.

    Returns (coeffs, residual_rms). coeffs is None on failure/insufficient points.
    """
    if len(points) < 6:
        return None, float('inf')
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    A = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)
        residual = np.sqrt(residuals[0] / len(points)) if len(residuals) > 0 else 0.0
        return coeffs, float(residual)
    except Exception:
        return None, float('inf')


def surface_volume_from_quadratic(
    points: np.ndarray,
    road_plane: np.ndarray,
    coeffs: np.ndarray,
    resolution: float = 0.01,
) -> Tuple[float, float]:
    """Integrate depth over pothole boundary to estimate surface-based volume and max depth.

    Returns (volume_m3, max_depth_m). If computation fails, returns (0.0, 0.0).
    """
    try:
        if coeffs is None or len(points) < 3:
            return 0.0, 0.0

        x_min, y_min = points[:, :2].min(axis=0)
        x_max, y_max = points[:, :2].max(axis=0)
        margin = 0.05
        x_min -= margin; x_max += margin
        y_min -= margin; y_max += margin

        x_grid = np.arange(x_min, x_max, resolution)
        y_grid = np.arange(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)

        a, b, c, d, e, f = coeffs
        z_surface = a*xx**2 + b*yy**2 + c*xx*yy + d*xx + e*yy + f
        plane_a, plane_b, plane_c, plane_d = road_plane
        z_road = -(plane_a*xx + plane_b*yy + plane_d) / (plane_c + 1e-12)
        depths_grid = z_road - z_surface

        from scipy.spatial import Delaunay
        hull = Delaunay(points[:, :2])
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        mask = hull.find_simplex(grid_points) >= 0
        mask = mask.reshape(xx.shape)

        depths_grid[~mask] = 0
        depths_grid[depths_grid < 0] = 0

        volume = float(np.sum(depths_grid) * (resolution**2))
        max_depth = float(np.max(depths_grid))
        return volume, max_depth
    except Exception:
        return 0.0, 0.0


def surface_depth_grid(
    points: np.ndarray,
    road_plane: np.ndarray,
    coeffs: np.ndarray,
    resolution: float = 0.01,
):
    """Compute (xx, yy, depths_grid, mask) for visualizing surface-based depths.

    Returns None on failure.
    """
    try:
        if coeffs is None or len(points) < 3:
            return None
        x_min, y_min = points[:, :2].min(axis=0)
        x_max, y_max = points[:, :2].max(axis=0)
        margin = 0.05
        x_min -= margin; x_max += margin
        y_min -= margin; y_max += margin

        x_grid = np.arange(x_min, x_max, resolution)
        y_grid = np.arange(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)

        a, b, c, d, e, f = coeffs
        z_surface = a*xx**2 + b*yy**2 + c*xx*yy + d*xx + e*yy + f
        pa, pb, pc, pd = road_plane
        z_road = -(pa*xx + pb*yy + pd) / (pc + 1e-12)
        depths_grid = z_road - z_surface

        from scipy.spatial import Delaunay
        hull = Delaunay(points[:, :2])
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        mask = hull.find_simplex(grid_points) >= 0
        mask = mask.reshape(xx.shape)

        depths_grid[~mask] = 0
        depths_grid[depths_grid < 0] = 0

        return xx, yy, depths_grid, mask
    except Exception:
        return None


def per_pothole_summary(
    points: np.ndarray,
    depths: np.ndarray,
    road_plane: Optional[np.ndarray] = None,
    compute_surface: bool = False,
) -> Dict:
    """Produce a compact set of metrics for a pothole cluster.

    Includes basic depth stats, convex-hull area/volume, and optionally
    quadratic-surface-based volume/max depth.
    """
    summary: Dict = {
        "points": int(len(points)),
        "max_depth": float(depths.max()) if len(depths) else 0.0,
        "mean_depth": float(depths.mean()) if len(depths) else 0.0,
        "median_depth": float(np.median(depths)) if len(depths) else 0.0,
    }
    hull_area, hull_obj = convex_hull_area(points[:, :2])
    summary["hull_area"] = hull_area
    summary["simple_volume"] = float(hull_area * (depths.mean() if len(depths) else 0.0))
    # Optional quadratic fit and surface-based volume
    if compute_surface and road_plane is not None:
        coeffs, residual = fit_quadratic_surface(points)
        summary["quad_residual"] = float(residual)
        if coeffs is not None:
            vol_surf, max_depth_surf = surface_volume_from_quadratic(points, road_plane, coeffs)
            summary["surface_volume"] = float(vol_surf)
            summary["surface_max_depth"] = float(max_depth_surf)
            if summary["simple_volume"] > 0:
                summary["volume_ratio_surface_over_convex"] = float(vol_surf / summary["simple_volume"])
    
    return summary


