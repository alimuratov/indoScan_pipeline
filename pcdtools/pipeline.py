from __future__ import annotations

import os
import numpy as np
from typing import Optional, Dict, Final

from .io import read_point_cloud, classify_points_by_color
from .plane import segment_plane_road, compute_depths_from_plane, filter_pothole_depths
from .cluster import dbscan_labels
from .analysis import per_pothole_summary, fit_quadratic_surface, surface_depth_grid
from .visualize import save_hull_plot, save_depth_heatmap, visualize_plane_and_potholes


# Constants
EPS_MAX: Final[float] = 1e6


def run_geometry_pipeline(
    pcd_path: str,
    eps: float = 0.1,
    summary_only: bool = False,
    save_hull_2d: bool = False,
    hull_plot_path: Optional[str] = None,
    aggregate_all: bool = False,
    visualize_3d: bool = False,
    save_surface_heatmap: bool = False,
    surface_heatmap_path: Optional[str] = None,
) -> None:
    """End-to-end pothole geometry pipeline.

    Steps:
      1) Read the point cloud and separate pothole vs road points (color-based).
      2) Fit the road plane with RANSAC and compute signed distances.
      3) Keep pothole points (below plane), convert depths to positive.
      4) Optionally cluster pothole points; compute per-cluster summaries.
      5) Optionally visualize hulls/3D scene and save surface depth heatmaps.
    """
    pcd = read_point_cloud(pcd_path)
    pothole_points, road_points = classify_points_by_color(pcd)

    if len(pcd.points) < 1:
        print("Empty point cloud")
        return
    if len(pothole_points) < 1:
        print("No pothole points found")
        return

    plane_model, inlier_mask = segment_plane_road(pcd)
    signed_depths = compute_depths_from_plane(pothole_points, plane_model)
    filtered_points, depths = filter_pothole_depths(pothole_points, signed_depths)
    if len(filtered_points) == 0:
        print("No pothole points below the road plane")
        return

    if summary_only and not aggregate_all:
        # Single-pothole style summary
        summary = per_pothole_summary(filtered_points, depths, plane_model, compute_surface=True)
        print("\n  Pothole-A:")
        print(f"    Points: {summary['points']}")
        print("    Basic depth analysis:")
        print(f"      Max depth: {summary['max_depth']:.4f} m")
        print(f"      Mean depth: {summary['mean_depth']:.4f} m")
        print(f"      Median depth: {summary['median_depth']:.4f} m")
        print("    Convex hull approximation:")
        print(f"      Area: {summary['hull_area']:.4f} m²")
        print(f"      Volume: {summary['simple_volume']:.6f} m³")

        if save_hull_2d:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(filtered_points[:, :2]) if len(filtered_points) >= 3 else None
            if hull is not None:
                out_path = hull_plot_path or "hull_2d.png"
                save_hull_plot(filtered_points[:, :2], hull, out_path, title=f"Hull (Area={summary['hull_area']:.4f} m²)")
                print(f"      Saved 2D hull plot -> {out_path}")
        # Optional surface heatmap for single cluster
        if save_surface_heatmap and 'surface_volume' in summary:
            coeffs, _ = fit_quadratic_surface(filtered_points)
            grid = surface_depth_grid(filtered_points, plane_model, coeffs)
            if grid is not None:
                xx, yy, depths_grid, mask = grid
                out_path = surface_heatmap_path or "surface_depth_heatmap.png"
                save_depth_heatmap(xx, yy, depths_grid, mask, out_path)
                print(f"      Saved surface depth heatmap -> {out_path}")
        if visualize_3d:
            try:
                pcd = read_point_cloud(pcd_path)
                visualize_plane_and_potholes(pcd, plane_model, filtered_points, None, None)
            except Exception:
                pass
        return

    labels, n_clusters = dbscan_labels(filtered_points, eps=eps)

    # If no clusters found, use max eps
    if n_clusters == 0:
        labels, n_clusters = dbscan_labels(filtered_points, eps=EPS_MAX)

    print("\nPothole Analysis Results")
    print(f"Plane model (a,b,c,d): {plane_model}")
    print(f"Road inliers: {inlier_mask.sum()} / {len(pcd.points)}")
    print(f"Pothole points: {len(pothole_points)}")
    print(f"Points below plane (actual potholes): {len(filtered_points)}")
    print(f"Number of potholes detected: {n_clusters}")

    summaries: Dict[int, Dict] = {}
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        pts = filtered_points[cluster_mask]
        dps = depths[cluster_mask]
        summary = per_pothole_summary(pts, dps, plane_model, compute_surface=True)
        summaries[cluster_id] = summary

        print(f"\n  Pothole-{chr(65+cluster_id) if cluster_id < 26 else cluster_id+1}:")
        print(f"    Points: {summary['points']}")
        print("    Basic depth analysis:")
        print(f"      Max depth: {summary['max_depth']:.4f} m")
        print(f"      Mean depth: {summary['mean_depth']:.4f} m")
        print(f"      Median depth: {summary['median_depth']:.4f} m")
        print("    Convex hull approximation:")
        print(f"      Area: {summary['hull_area']:.4f} m²")
        print(f"      Volume: {summary['simple_volume']:.6f} m³")
        if 'surface_volume' in summary:
            print("    Quadratic surface fitting:")
            print(f"      Surface-based volume: {summary['surface_volume']:.6f} m³")
            print(f"      Surface-based max depth: {summary.get('surface_max_depth', 0.0):.4f} m")
            vr = summary.get('volume_ratio_surface_over_convex', None)
            if vr is not None:
                print(f"      Volume ratio (surface/convex): {vr:.2f}")

        if save_hull_2d and len(pts) >= 3:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts[:, :2])
            base, ext = os.path.splitext(hull_plot_path) if hull_plot_path else ("hull", ".png")
            out_path = f"{base}_cluster_{cluster_id}{ext}"
            save_hull_plot(pts[:, :2], hull, out_path, title=f"Pothole-{cluster_id} (Area={summary['hull_area']:.4f} m²)")
            print(f"      Saved 2D hull plot -> {out_path}")
        if save_surface_heatmap and 'surface_volume' in summary:
            coeffs, _ = fit_quadratic_surface(pts)
            grid = surface_depth_grid(pts, plane_model, coeffs)
            if grid is not None:
                xx, yy, depths_grid, mask = grid
                base, ext = os.path.splitext(surface_heatmap_path) if surface_heatmap_path else ("surface_heatmap", ".png")
                out_path = f"{base}_cluster_{cluster_id}{ext}"
                save_depth_heatmap(xx, yy, depths_grid, mask, out_path)
                print(f"      Saved surface depth heatmap -> {out_path}")

    # Overall stats
    print(f"\nOverall statistics (all potholes):")
    print(f"  Max depth: {depths.max():.4f} m")
    print(f"  Mean depth: {depths.mean():.4f} m")
    print(f"  Median depth: {np.median(depths):.4f} m")

    if aggregate_all and n_clusters > 0:
        total_area = sum(s.get("hull_area", 0.0) for s in summaries.values())
        total_volume = sum(s.get("simple_volume", 0.0) for s in summaries.values())
        avg_of_means = np.mean([s.get("mean_depth", 0.0) for s in summaries.values()])
        print(f"\nAggregated across clusters:")
        print(f"  Sum of areas (convex hull): {total_area:.6f} m²")
        print(f"  Sum of volumes (convex hull approx.): {total_volume:.6f} m³")
        print(f"  Average of mean depths: {avg_of_means:.6f} m")

    if visualize_3d:
        try:
            pcd = read_point_cloud(pcd_path)
            visualize_plane_and_potholes(pcd, plane_model, filtered_points, labels, n_clusters)
        except Exception:
            pass


