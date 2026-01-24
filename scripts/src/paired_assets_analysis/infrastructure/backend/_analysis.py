"""Pothole metrics and analysis utilities."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


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


def _z_on_plane(plane: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Evaluate plane ax + by + cz + d = 0 as z(x, y) on XY points.

    Raises if the plane is nearly vertical (|c| ~ 0).
    """
    a, b, c, d = plane
    if abs(float(c)) < 1e-12:
        raise ValueError("Plane nearly vertical; cannot express z(x, y)")
    return -(a * xy[:, 0] + b * xy[:, 1] + d) / c


def tin_volume_over_points(
    points: np.ndarray,
    road_plane: np.ndarray,
) -> Tuple[float, float]:
    """Compute volume under the road plane using a TIN over the pothole points.

    Triangulates XY and, for each triangle, adds area_xy * mean(depths at vertices),
    where depth = z_road(x, y) - z_point, clamped to >= 0.

    Returns (volume, max_depth) with units consistent with inputs (e.g., m^3, m).
    """
    if points.shape[0] < 3:
        return 0.0, 0.0

    xy = points[:, :2]
    z = points[:, 2]

    # Triangulate
    try:
        from scipy.spatial import Delaunay

        tri = Delaunay(xy)
        triangles = np.asarray(tri.simplices, dtype=np.int32)
    except Exception:
        try:
            import matplotlib.tri as mtri

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
        d0 = max(d0, 0.0)
        d1 = max(d1, 0.0)
        d2 = max(d2, 0.0)
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
    TIN-based (Delaunay) volume/max depth.
    """
    summary: Dict = {
        "points": int(len(points)),
        "max_depth": float(depths.max()) if len(depths) else 0.0,
        "mean_depth": float(depths.mean()) if len(depths) else 0.0,
        "median_depth": float(np.median(depths)) if len(depths) else 0.0,
    }
    hull_area, hull_obj = convex_hull_area(points[:, :2])
    summary["hull_area"] = hull_area
    summary["simple_volume"] = float(
        hull_area * (depths.mean() if len(depths) else 0.0)
    )

    # TIN-based (Delaunay) volume over the pothole footprint relative to the road plane
    if compute_surface and road_plane is not None:
        vol_tin, max_depth_tin = tin_volume_over_points(points, road_plane)
        summary["delaunay_volume"] = float(vol_tin)
        if summary["simple_volume"] > 0:
            summary["volume_ratio_delaunay_over_convex"] = float(
                vol_tin / summary["simple_volume"]
            )

    return summary

