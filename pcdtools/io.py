from __future__ import annotations

import open3d as o3d
import numpy as np


def read_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Read a point cloud from disk using Open3D.

    Supported formats include PCD/PLY/XYZ depending on Open3D compilation.
    """
    return o3d.io.read_point_cloud(path)


def classify_points_by_color(pcd: o3d.geometry.PointCloud, red_threshold: float = 0.7):
    """Split points into pothole vs road using a simple red-channel heuristic.

    - Pothole points: R > red_threshold and G,B < 0.3
    - Road points: complement
    Returns (pothole_points, road_points) as numpy arrays of shape (N,3).
    If no color is present, all points are treated as road and potholes empty.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if colors.size == 0:
        return np.array([]), points

    red_mask = (colors[:, 0] > red_threshold) & (colors[:, 1] < 0.3) & (colors[:, 2] < 0.3)
    road_mask = ~red_mask
    pothole_points = points[red_mask]
    road_points = points[road_mask]
    return pothole_points, road_points


