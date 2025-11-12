"""Output presentation helpers for pipeline results.

Separates printing/formatting logic from core pipeline orchestration.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np


def cluster_name(cluster_id: Optional[int]) -> str:
    """Human-friendly cluster label."""
    if cluster_id is None:
        return "Pothole-A"
    return f"Pothole-{chr(65+cluster_id) if cluster_id < 26 else cluster_id+1}"


def print_cluster_summary(summary: Dict[str, Any], cluster_id: int) -> None:
    """Print formatted summary for a single cluster."""
    print(f"\n  {cluster_name(cluster_id)}:")
    print(f"    Points: {summary['points']}")
    print("    Basic depth analysis:")
    print(f"      Max depth: {summary['max_depth']:.4f} m")
    print(f"      Mean depth: {summary['mean_depth']:.4f} m")
    print(f"      Median depth: {summary['median_depth']:.4f} m")
    print("    Convex hull approximation:")
    print(f"      Area: {summary['hull_area']:.4f} m²")
    print(f"      Volume: {summary['simple_volume']:.6f} m³")
    if 'delaunay_volume' in summary:
        print("    Delaunay-based volume:")
        print(f"      Delaunay volume: {summary['delaunay_volume']:.6f} m³")
        vr = summary.get('volume_ratio_delaunay_over_convex', None)
        if vr is not None:
            print(f"      Volume ratio (surface/convex): {vr:.2f}")


def print_pipeline_header(
    plane_model: np.ndarray,
    inlier_mask: np.ndarray,
    pothole_points: np.ndarray,
    filtered_points: np.ndarray,
    n_clusters: int,
    total_points: int
) -> None:
    """Print pipeline analysis header with counts."""
    print("\nPothole Analysis Results")
    print(f"Plane model (a,b,c,d): {plane_model}")
    print(f"Road inliers: {inlier_mask.sum()} / {total_points}")
    print(f"Pothole points: {len(pothole_points)}")
    print(f"Points below plane (actual potholes): {len(filtered_points)}")
    print(f"Number of potholes detected: {n_clusters}")


def print_overall_stats(depths: np.ndarray) -> None:
    """Print overall depth statistics."""
    print(f"\nOverall statistics (all potholes):")
    print(f"  Max depth: {depths.max():.4f} m")
    print(f"  Mean depth: {depths.mean():.4f} m")
    print(f"  Median depth: {np.median(depths):.4f} m")


def print_aggregated_stats(summaries: Dict[int, Dict]) -> None:
    """Print aggregated metrics across all clusters."""
    if len(summaries) == 0:
        return
    total_area = sum(s.get("hull_area", 0.0) for s in summaries.values())
    total_volume = sum(s.get("simple_volume", 0.0) for s in summaries.values())
    avg_of_means = np.mean([s.get("mean_depth", 0.0)
                           for s in summaries.values()])
    print(f"\nAggregated across clusters:")
    print(f"  Sum of areas (convex hull): {total_area:.6f} m²")
    print(f"  Sum of volumes (convex hull approx.): {total_volume:.6f} m³")
    print(f"  Average of mean depths: {avg_of_means:.6f} m")
