from __future__ import annotations

import numpy as np
import open3d as o3d
from typing import Tuple


def segment_plane_road(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the dominant plane (road) in the cloud via RANSAC.

    Returns (plane_model, inlier_mask) where plane_model is (a,b,c,d) for ax+by+cz+d=0
    and inlier_mask is a boolean array for road points.
    """
    if len(pcd.points) == 0:
        raise ValueError("Empty point cloud")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations
    )
    pts = np.asarray(pcd.points)
    inlier_mask = np.zeros(len(pts), dtype=bool)
    inlier_mask[np.array(inliers, dtype=int)] = True
    return np.array(plane_model), inlier_mask


def compute_depths_from_plane(points: np.ndarray, plane_model: np.ndarray) -> np.ndarray:
    """Compute signed distance of points to plane.

    Positive above plane, negative below plane.
    """
    if points.size == 0:
        return np.zeros((0,), dtype=float)
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=float)
    n_norm = np.linalg.norm(n) + 1e-12
    depths = (points @ n + d) / n_norm
    return depths


def filter_pothole_depths(points: np.ndarray, depths: np.ndarray, threshold: float = 0.0):
    """Keep only points below the road plane and return positive depths."""
    mask = depths < threshold
    filtered_points = points[mask]
    filtered_depths = np.abs(depths[mask])
    return filtered_points, filtered_depths


