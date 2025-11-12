from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np
from typing import Optional, Dict, Final, Any

import open3d as o3d
from .io import read_point_cloud, classify_points_by_color
from .plane import segment_plane_road, compute_depths_from_plane, filter_pothole_depths
from .cluster import dbscan_labels
from .analysis import per_pothole_summary
from .visualize import visualize_plane_and_potholes
from .strategies import TrianglePruningStrategy, PerimeterOnlyPruningStrategy
from .presenters import (
    cluster_name,
    print_cluster_summary,
    print_pipeline_header,
    print_overall_stats,
    print_aggregated_stats,
)
from .exporters import export_artifacts_for_cluster


# Constants
EPS_MAX: Final[float] = 1e6

"""Dataclasses for visualization options and output path configuration."""


class PipelineBuilder:
    """Fluent API builder for configuring and running the pothole geometry pipeline.

    Example:
        result = (PipelineBuilder("pothole.pcd")
            .with_clustering(eps=0.2)
            .with_hull_plot("hull.png")
            .with_tin_mesh("tin.ply")
            .with_3d_visualization()
            .run())
    """

    def __init__(self, pcd_path: str):
        """Initialize builder with input point cloud path."""
        self.pcd_path = pcd_path
        self.eps = 0.1
        self.summary_only = False
        self.aggregate_all = False
        self.plane_extent_margin = 0.1
        self.plane_with_hole_grid_res = 64
        self._viz_config = VisualizationConfig()
        self._out_config = OutputPathsConfig()
        self._triangle_pruning: Optional[TrianglePruningStrategy] = None

    # Clustering configuration
    def with_clustering(self, eps: float = 0.1) -> "PipelineBuilder":
        """Configure DBSCAN clustering epsilon."""
        self.eps = eps
        self.summary_only = False
        return self

    def without_clustering(self) -> "PipelineBuilder":
        """Disable clustering (single pothole mode)."""
        self.summary_only = True
        return self

    def with_aggregation(self) -> "PipelineBuilder":
        """Enable aggregated metrics across clusters."""
        self.aggregate_all = True
        return self

    # Visualization configuration
    def with_3d_visualization(self, enable: bool = True) -> "PipelineBuilder":
        """Enable/disable interactive 3D visualization."""
        self._viz_config.visualize_3d = enable
        return self

    def with_overlay_visualization(self, enable: bool = True) -> "PipelineBuilder":
        """Enable/disable mesh overlay visualization."""
        self._viz_config.visualize_overlay = enable
        return self

    def with_hull_plot(self, path: Optional[str] = "hull.png") -> "PipelineBuilder":
        """Save 2D convex hull plot."""
        self._viz_config.save_hull_2d = True
        self._out_config.hull_plot_path = path
        return self

    def with_surface_heatmap(self, path: Optional[str] = "heatmap.png") -> "PipelineBuilder":
        """Save surface depth heatmap."""
        self._viz_config.save_surface_heatmap = True
        self._out_config.surface_heatmap_path = path
        return self

    def with_z_exaggeration(self, factor: float) -> "PipelineBuilder":
        """Set Z-axis exaggeration factor for meshes."""
        self._viz_config.tin_z_exaggeration = factor
        return self

    def with_pothole_plane_helper(self, enable: bool = True) -> "PipelineBuilder":
        """Save helper mesh with fitted plane and pothole points."""
        self._viz_config.save_pothole_points_with_fitted_plane = enable
        return self

    # Output paths configuration
    def with_tin_mesh(self, path: str) -> "PipelineBuilder":
        """Save TIN mesh to specified path."""
        self._out_config.save_tin_mesh_path = path
        return self

    def with_tin_points(self, path: str) -> "PipelineBuilder":
        """Save TIN vertices as point cloud."""
        self._out_config.save_tin_points_path = path
        return self

    def with_complete_mesh(self, path: str) -> "PipelineBuilder":
        """Save complete pothole mesh (TIN + walls)."""
        self._out_config.save_complete_pothole_mesh_path = path
        return self

    def with_plane_mesh(self, path: str) -> "PipelineBuilder":
        """Save standalone road plane mesh."""
        self._out_config.save_plane_mesh_path = path
        return self

    def with_plane_with_hole(self, path: str) -> "PipelineBuilder":
        """Save road plane mesh with pothole cut out."""
        self._out_config.save_plane_with_hole_mesh_path = path
        return self

    def with_combined_mesh(self, path: str) -> "PipelineBuilder":
        """Save combined mesh (pothole + plane-with-hole)."""
        self._out_config.save_combined_mesh_path = path
        return self

    def with_overlay_mesh(self, path: str) -> "PipelineBuilder":
        """Save overlay mesh (combined + input PCD as spheres)."""
        self._out_config.save_overlay_with_pcd_mesh_path = path
        return self

    def with_delaunay_contrib_2d(self, path: str) -> "PipelineBuilder":
        """Save 2D Delaunay contribution visualization."""
        self._out_config.save_delaunay_contrib_2d_path = path
        return self

    def with_delaunay_contrib_3d(self, path: str) -> "PipelineBuilder":
        """Save 3-view 3D Delaunay contribution visualization."""
        self._out_config.save_delaunay_contrib_3d_multi_path = path
        return self

    # Advanced configuration
    def with_pruning_strategy(self, strategy: TrianglePruningStrategy) -> "PipelineBuilder":
        """Set custom triangle pruning strategy."""
        self._triangle_pruning = strategy
        return self

    def with_plane_extent_margin(self, margin: float) -> "PipelineBuilder":
        """Set margin around pothole extent for plane mesh."""
        self.plane_extent_margin = margin
        return self

    def with_plane_grid_resolution(self, resolution: int) -> "PipelineBuilder":
        """Set grid resolution for plane-with-hole mesh."""
        self.plane_with_hole_grid_res = resolution
        return self

    def with_pcd_sphere_params(
        self,
        radius: float = 0.02,
        max_points: int = 5000,
        resolution: int = 3,
        seed: int = 0
    ) -> "PipelineBuilder":
        """Configure PCD-to-spheres conversion parameters for overlay."""
        self._viz_config.pcd_sphere_radius = radius
        self._viz_config.pcd_max_points = max_points
        self._viz_config.pcd_sphere_resolution = resolution
        self._viz_config.pcd_random_seed = seed
        return self

    def with_visualization_config(self, config: VisualizationConfig) -> "PipelineBuilder":
        """Set entire visualization config at once."""
        self._viz_config = config
        return self

    def with_output_paths_config(self, config: OutputPathsConfig) -> "PipelineBuilder":
        """Set entire output paths config at once."""
        self._out_config = config
        return self

    # Execution
    def run(self) -> None:
        """Execute the configured pipeline."""
        run_geometry_pipeline(
            pcd_path=self.pcd_path,
            eps=self.eps,
            summary_only=self.summary_only,
            aggregate_all=self.aggregate_all,
            plane_extent_margin=self.plane_extent_margin,
            plane_with_hole_grid_res=self.plane_with_hole_grid_res,
            visualization=self._viz_config,
            output_paths=self._out_config,
            triangle_pruning=self._triangle_pruning,
        )

    def analyze(self) -> Dict[str, Any]:
        """Execute analysis and return structured results instead of printing."""
        return analyze_pothole_geometry(
            pcd_path=self.pcd_path,
            eps=self.eps,
            summary_only=self.summary_only,
            aggregate_all=self.aggregate_all,
        )


@dataclass
class VisualizationConfig:
    visualize_3d: bool = False
    visualize_overlay: bool = False
    save_hull_2d: bool = False
    save_surface_heatmap: bool = False
    tin_z_exaggeration: float = 1.0
    include_input_pcd_in_combined: bool = False
    pcd_sphere_radius: float = 0.02
    pcd_max_points: int = 5000
    pcd_sphere_resolution: int = 3
    pcd_random_seed: int = 0
    save_pothole_points_with_fitted_plane: bool = False


@dataclass
class OutputPathsConfig:
    hull_plot_path: Optional[str] = None
    surface_heatmap_path: Optional[str] = None
    save_delaunay_contrib_2d_path: Optional[str] = None
    save_delaunay_contrib_3d_multi_path: Optional[str] = None
    save_tin_mesh_path: Optional[str] = None
    save_tin_points_path: Optional[str] = None
    save_complete_pothole_mesh_path: Optional[str] = None
    save_plane_with_hole_mesh_path: Optional[str] = None
    save_combined_mesh_path: Optional[str] = None
    save_overlay_with_pcd_mesh_path: Optional[str] = None
    save_plane_mesh_path: Optional[str] = None


 


def run_geometry_pipeline(
    pcd_path: str,
    eps: float = 0.1,
    summary_only: bool = False,
    aggregate_all: bool = False,
    plane_extent_margin: float = 0.1,
    plane_with_hole_grid_res: int = 64,
    visualization: Optional[VisualizationConfig] = None,
    output_paths: Optional[OutputPathsConfig] = None,
    triangle_pruning: Optional[TrianglePruningStrategy] = None,
) -> None:
    """End-to-end pothole geometry pipeline.

    Steps:
      1) Read the PCD and split road vs pothole points by color.
      2) Fit the road plane (RANSAC)
      3) Compute signed depths to the plane.
      4) Keep pothole points (below plane) and convert depths to positive.
      5) Optionally cluster pothole points (DBSCAN); compute per‑cluster summaries
         including convex‑hull area/volume and Delaunay (TIN) volume.
      6) Build a pruned Delaunay triangulation (flatness + alpha filters) over
         pothole XY and use it for volume integration and contribution visuals.
      7) Optionally save artifacts: 2D/3‑view 3D Delaunay contribution images,
         TIN mesh/vertices, complete pothole mesh, plane‑with‑hole mesh,
         combined mesh, overlay (combined + input PCD spheres), and a standalone
         plane mesh sized to the pothole extent.
      8) Optionally save a 2D hull plot and a quadratic‑surface depth heatmap;
         optionally open interactive 3D views.
      9) Print per‑cluster and overall statistics; optionally aggregate metrics
         across clusters.

    Args:
      pcd_path: Absolute path to an input `.pcd` containing road + pothole points.
        Pothole points should be red‑tinted so they can be separated by color.
      eps: DBSCAN epsilon (meters) for clustering pothole points.
      summary_only: If True, compute a single summary without clustering.
      aggregate_all: If True, print aggregated metrics across clusters.
      plane_extent_margin: Extra margin (meters) around pothole extent when
        sizing the standalone plane mesh.
      plane_with_hole_grid_res: Grid resolution for building the plane‑with‑hole
        mesh.
      visualization: VisualizationConfig with all visualization toggles and
        mesh‑visual related parameters.
      output_paths: OutputPathsConfig with all optional output destinations.

    Returns:
      None. Prints summaries to stdout and writes artifacts only when output paths
      are provided. Per‑cluster exports automatically get `_cluster_{id}` suffixes.

    Raises:
      None directly. I/O and mesh‑building errors are caught and logged; missing
      or empty inputs result in early returns with console messages.
    """
    # Normalize inputs into config objects
    viz_cfg = visualization or VisualizationConfig()
    out_cfg = output_paths or OutputPathsConfig()

    # Load and validate data
    pcd = read_point_cloud(pcd_path)
    pothole_points, road_points = classify_points_by_color(pcd)

    if len(pcd.points) < 1:
        print("Empty point cloud")
        return
    if len(pothole_points) < 1:
        print("No pothole points found")
        return

    # Fit plane and filter pothole points
    plane_model, inlier_mask = segment_plane_road(pcd)
    signed_depths = compute_depths_from_plane(pothole_points, plane_model)
    filtered_points, depths = filter_pothole_depths(
        pothole_points, signed_depths)

    if len(filtered_points) == 0:
        print("No pothole points below the road plane")
        return

    # Cluster pothole points
    if summary_only:
        labels = np.zeros(len(filtered_points), dtype=int)
        n_clusters = 1
    else:
        labels, n_clusters = dbscan_labels(filtered_points, eps=eps)
        if n_clusters == 0:
            labels, n_clusters = dbscan_labels(filtered_points, eps=EPS_MAX)

    # Print header with counts
    print_pipeline_header(plane_model, inlier_mask, pothole_points,
                          filtered_points, n_clusters, len(pcd.points))

    summaries: Dict[int, Dict] = {}
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        pts = filtered_points[cluster_mask]

        dps = depths[cluster_mask]
        summary = per_pothole_summary(
            pts, dps, plane_model, compute_surface=True)
        summaries[cluster_id] = summary

        print_cluster_summary(summary, cluster_id)

        export_artifacts_for_cluster(
            pts,
            plane_model,
            cluster_id=cluster_id,
            has_surface=('delaunay_volume' in summary),
            viz_cfg=viz_cfg,
            out_cfg=out_cfg,
            pcd=pcd,
            plane_extent_margin=plane_extent_margin,
            plane_with_hole_grid_res=plane_with_hole_grid_res,
            pcd_path=pcd_path,
            triangle_pruning=triangle_pruning,
        )

    # Overall stats
    print_overall_stats(depths)

    if aggregate_all and n_clusters > 0:
        print_aggregated_stats(summaries)

    if viz_cfg.visualize_3d:
        try:
            pcd = read_point_cloud(pcd_path)
            visualize_plane_and_potholes(
                pcd, plane_model, filtered_points, labels, n_clusters)
        except Exception:
            pass


def analyze_pothole_geometry(
    pcd_path: str,
    eps: float = 0.1,
    summary_only: bool = False,
    aggregate_all: bool = False,
) -> Dict[str, Any]:
    """Programmatic API: return structured pothole geometry results.

    Returns a dictionary with keys:
    - status: "ok" | "empty_cloud" | "no_pothole_points" | "no_points_below_plane"
    - plane_model: list[float] (a,b,c,d) if available
    - n_clusters: int (1 for summary_only)
    - clusters: list of per-pothole summaries (see analysis.per_pothole_summary)
    - overall: dict with overall stats (max_depth, mean_depth, median_depth)
    - aggregate: dict with aggregated metrics when aggregate_all=True (optional)
    """
    result: Dict[str, Any] = {
        "status": "ok",
        "plane_model": None,
        "n_clusters": 0,
        "clusters": [],
        "overall": None,
    }

    pcd = read_point_cloud(pcd_path)
    pothole_points, road_points = classify_points_by_color(pcd)

    if len(pcd.points) < 1:
        result["status"] = "empty_cloud"
        return result
    if len(pothole_points) < 1:
        result["status"] = "no_pothole_points"
        return result

    plane_model, inlier_mask = segment_plane_road(pcd)
    result["plane_model"] = [float(x) for x in plane_model]

    signed_depths = compute_depths_from_plane(pothole_points, plane_model)
    filtered_points, depths = filter_pothole_depths(
        pothole_points, signed_depths)

    if len(filtered_points) == 0:
        result["status"] = "no_points_below_plane"
        return result

    # Cluster pothole points
    if summary_only:
        labels = np.zeros(len(filtered_points), dtype=int)
        n_clusters = 1
    else:
        labels, n_clusters = dbscan_labels(filtered_points, eps=eps)
        if n_clusters == 0:
            labels, n_clusters = dbscan_labels(filtered_points, eps=EPS_MAX)

    summaries: Dict[int, Dict] = {}
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        pts = filtered_points[cluster_mask]
        dps = depths[cluster_mask]
        summary = per_pothole_summary(
            pts, dps, plane_model, compute_surface=True)
        summaries[cluster_id] = summary

    result["n_clusters"] = int(n_clusters)
    result["clusters"] = [summaries[i] for i in range(n_clusters)]
    result["overall"] = {
        "max_depth": float(depths.max()),
        "mean_depth": float(depths.mean()),
        "median_depth": float(np.median(depths)),
    }

    if aggregate_all and n_clusters > 0:
        total_area = sum(s.get("hull_area", 0.0) for s in summaries.values())
        total_volume = sum(s.get("simple_volume", 0.0)
                           for s in summaries.values())
        avg_of_means = float(
            np.mean([s.get("mean_depth", 0.0) for s in summaries.values()]))
        result["aggregate"] = {
            "sum_area_hull": float(total_area),
            "sum_volume_hull": float(total_volume),
            "average_of_mean_depths": avg_of_means,
        }

    return result
