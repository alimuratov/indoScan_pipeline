from __future__ import annotations

import open3d as o3d
import numpy as np


def read_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Read a point cloud from disk using Open3D.

    Supported formats include PCD/PLY/XYZ depending on Open3D compilation.
    """
    return o3d.io.read_point_cloud(path)


def write_point_cloud_from_arrays(
    path: str,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """Write a point cloud to disk from numpy arrays using Open3D.

    Args:
        path: Output file path (e.g., .pcd, .ply). Extension determines format.
        points: Array of shape (N, 3) with XYZ coordinates.
        colors: Optional array of shape (N, 3) with RGB in [0, 1].
        write_ascii: If True, write in ASCII format when supported.
        compressed: If True, write in compressed binary format when supported.

    Raises:
        ValueError: If input shapes are invalid or sizes mismatch.
        RuntimeError: If the point cloud cannot be written.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors is not None:
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError("colors must have shape (N, 3) when provided")
        if colors.shape[0] != points.shape[0]:
            raise ValueError("colors and points must have the same number of rows (N)")

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None and colors.size > 0:
        pc.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))

    ok = o3d.io.write_point_cloud(
        path,
        pc,
        print_progress=False,
    )
    if not ok:
        raise RuntimeError(f"Failed to write point cloud to {path}")


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



def remove_pothole_points(
    pcd: o3d.geometry.PointCloud,
    *,
    red_threshold: float = 0.7,
    green_max: float = 0.3,
    blue_max: float = 0.3,
):
    """Return a new point cloud without red-colored pothole points and the removed set.

    Returns (road_pcd, pothole_points, pothole_colors).
    If no colors are present, returns (pcd copy, empty arrays, None).
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # If there are no colors, nothing to remove by color
    if colors.size == 0 or points.size == 0:
        road_copy = o3d.geometry.PointCloud()
        road_copy.points = o3d.utility.Vector3dVector(points.copy())
        return road_copy, np.empty((0, 3), dtype=float), None

    red_mask = (colors[:, 0] > red_threshold) & (colors[:, 1] < green_max) & (colors[:, 2] < blue_max)
    keep_mask = ~red_mask

    road_points = points[keep_mask]
    road_colors = colors[keep_mask]
    pothole_points = points[red_mask]
    pothole_colors = colors[red_mask]

    road_pcd = o3d.geometry.PointCloud()
    road_pcd.points = o3d.utility.Vector3dVector(road_points)
    if road_colors.size > 0:
        road_pcd.colors = o3d.utility.Vector3dVector(road_colors.astype(float))

    return road_pcd, pothole_points, pothole_colors


def add_pothole_points_back(
    road_pcd: o3d.geometry.PointCloud,
    pothole_points: np.ndarray,
    pothole_colors: np.ndarray | None = None,
):
    """Combine road-only cloud with pothole points (and optional colors) into one cloud.

    If road_pcd has colors and pothole_colors is provided, colors are concatenated.
    If road_pcd has no colors but pothole_colors is provided, colors are set for all points.
    """
    road_points = np.asarray(road_pcd.points)
    combined_points = (
        road_points if pothole_points is None or pothole_points.size == 0 else np.vstack([road_points, pothole_points])
    )

    combined = o3d.geometry.PointCloud()
    combined.points = o3d.utility.Vector3dVector(combined_points)

    road_colors = np.asarray(road_pcd.colors)
    if road_colors.size > 0:
        if pothole_colors is not None and pothole_colors.size > 0:
            combined_colors = (
                road_colors
                if pothole_points is None or pothole_points.size == 0
                else np.vstack([road_colors, pothole_colors])
            )
            combined.colors = o3d.utility.Vector3dVector(combined_colors.astype(float))
        else:
            combined.colors = o3d.utility.Vector3dVector(road_colors.astype(float))
    elif pothole_colors is not None and pothole_colors.size > 0:
        # Road had no colors; assign colors only if sizes match
        num_combined = combined_points.shape[0]
        num_road = road_points.shape[0]
        if pothole_points is not None and pothole_points.size > 0 and (num_road + pothole_colors.shape[0]) == num_combined:
            pad = np.zeros((num_road, 3), dtype=float)
            combined_colors = np.vstack([pad, pothole_colors])
            combined.colors = o3d.utility.Vector3dVector(combined_colors.astype(float))

    return combined

