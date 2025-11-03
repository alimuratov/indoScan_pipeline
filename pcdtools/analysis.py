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

        # Create the integration grid
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


def _z_on_plane(plane: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Evaluate plane ax+by+cz+d=0 as z(x,y) on XY points.

    Raises if the plane is nearly vertical (|c| ~ 0).
    """
    a, b, c, d = plane
    if abs(float(c)) < 1e-12:
        raise ValueError("Plane nearly vertical; cannot express z(x,y)")
    return -(a * xy[:, 0] + b * xy[:, 1] + d) / c


def tin_volume_over_points(points: np.ndarray, road_plane: np.ndarray) -> tuple[float, float]:
    """Compute volume under the road plane using a TIN over the pothole points.

    Triangulates XY and, for each triangle, adds area_xy * mean(depths at vertices),
    where depth = z_road(x,y) - z_point, clamped to >= 0.

    Returns (volume, max_depth) with units consistent with inputs (e.g., m^3, m).
    """
    if points.shape[0] < 3:
        return 0.0, 0.0

    xy = points[:, :2]
    z = points[:, 2]

    # Triangulate
    try:
        from scipy.spatial import Delaunay  # type: ignore
        tri = Delaunay(xy)
        triangles = np.asarray(tri.simplices, dtype=np.int32)
    except Exception:
        try:
            import matplotlib.tri as mtri  # type: ignore
            triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
            triangles = np.asarray(triang.triangles, dtype=np.int32)
        except Exception:
            return 0.0, 0.0

    if triangles.size == 0:
        return 0.0, 0.0

    # Depths at vertices relative to road plane
    try:
        z_road = _z_on_plane(road_plane, xy)
    except Exception:
        return 0.0, 0.0
    depth = z_road - z

    vol = 0.0
    max_depth = float(np.max(np.maximum(depth, 0.0))) if depth.size else 0.0
    for i, j, k in triangles:
        v0, v1, v2 = xy[i], xy[j], xy[k]
        # XY area of the triangle
        area = 0.5 * abs(np.linalg.det(np.stack([v1 - v0, v2 - v0], axis=0)))
        d0, d1, d2 = depth[i], depth[j], depth[k]
        # Clamp to only integrate below-plane regions
        d0 = max(d0, 0.0); d1 = max(d1, 0.0); d2 = max(d2, 0.0)
        vol += area * (d0 + d1 + d2) / 3.0

    return float(vol), float(max_depth)


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
    # TIN-based (Delaunay) volume over the pothole footprint relative to the road plane
    if compute_surface and road_plane is not None:
        vol_tin, max_depth_tin = tin_volume_over_points(points, road_plane)
        summary["delaunay_volume"] = float(vol_tin)
        # Max depth is already provided in summary["max_depth"], so we skip a separate field
        if summary["simple_volume"] > 0:
            summary["volume_ratio_delaunay_over_convex"] = float(vol_tin / summary["simple_volume"])
    
    
    
    return summary


