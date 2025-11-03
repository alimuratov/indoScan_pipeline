from __future__ import annotations

import os
import numpy as np
from typing import Optional, Dict, Final, Any

import open3d as o3d
from .io import read_point_cloud, classify_points_by_color
from .plane import segment_plane_road, compute_depths_from_plane, filter_pothole_depths
from .cluster import dbscan_labels
from .analysis import per_pothole_summary, fit_quadratic_surface, surface_depth_grid
from .visualize import save_hull_plot, save_depth_heatmap, visualize_plane_and_potholes, plane_mesh_covering_cloud


# Constants
EPS_MAX: Final[float] = 1e6


def _get_output_path(base_path: str, cluster_id: Optional[int], default_ext: str = ".png") -> str:
    """Generate output path with optional cluster suffix."""
    if cluster_id is None:
        return base_path
    base, ext = os.path.splitext(base_path)
    return f"{base}_cluster_{cluster_id}{ext or default_ext}"


def _validate_points_for_export(points: np.ndarray, min_points: int = 3) -> bool:
    """Common validation for point-based exports."""
    return points.ndim == 2 and points.shape[1] == 3 and points.shape[0] >= min_points


def _build_pruned_delaunay_triangles(
    points: np.ndarray,
    *,
    edge_percentile: float = 99.0,
    circ_percentile: float = 99.0,
    min_circle_ratio: float = 0.005,
) -> np.ndarray:
    """Build Delaunay triangles with flatness and alpha pruning (consistent with TIN builder)."""
    if points.shape[0] < 3:
        return np.array([], dtype=np.int32).reshape(0, 3)
    
    try:
        from .visualize_delaunay_volume import delaunay_tris
        from .triangular_mesh_construction import _alpha_filter_triangles
        import matplotlib.tri as mtri
        
        xy = points[:, :2]
        tris = delaunay_tris(xy)
        
        # Flatness pruning
        if tris.size > 0:
            try:
                tri_obj = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
                analyzer = mtri.TriAnalyzer(tri_obj)
                flat_mask = analyzer.get_flat_tri_mask(min_circle_ratio=min_circle_ratio)
                tris = tris[~flat_mask]
            except Exception:
                pass
        
        # Alpha pruning
        if tris.size > 0:
            tris = _alpha_filter_triangles(xy, tris, edge_percentile=edge_percentile, circ_percentile=circ_percentile)
        
        return tris
    except Exception:
        return np.array([], dtype=np.int32).reshape(0, 3)


def _save_delaunay_contrib_images(
    points: np.ndarray,
    plane_model: np.ndarray,
    *,
    save_2d_path: Optional[str] = None,
    save_3d_path: Optional[str] = None,
    cluster_id: Optional[int] = None,
) -> None:
    """Save Delaunay contribution visualizations using pruned triangles."""
    if not (save_2d_path or save_3d_path) or points.shape[0] < 3:
        return
    
    try:
        from .visualize_delaunay_volume import triangle_contributions, plot_2d_contrib, plot_3d_contrib_multi
        
        tris = _build_pruned_delaunay_triangles(points)
        if tris.size == 0:
            return
        
        xy = points[:, :2]
        contrib, total = triangle_contributions(points, tris, tuple(plane_model))
        
        if save_2d_path:
            final_2d_path = _get_output_path(save_2d_path, cluster_id, default_ext=".png")
            plot_2d_contrib(xy, tris, contrib, final_2d_path, total)
            print(f"      Saved Delaunay 2D contrib -> {final_2d_path}")
        if save_3d_path:
            final_3d_path = _get_output_path(save_3d_path, cluster_id, default_ext=".png")
            plot_3d_contrib_multi(points, tris, contrib, final_3d_path)
            print(f"      Saved Delaunay 3-view 3D contrib -> {final_3d_path}")
    except Exception:
        pass


def _save_tin_mesh_and_points(
    points: np.ndarray,
    plane_model: np.ndarray,
    *,
    mesh_path: Optional[str] = None,
    points_path: Optional[str] = None,
    complete_mesh_path: Optional[str] = None,
    z_exaggeration: float = 1.0,
    cluster_id: Optional[int] = None,
) -> None:
    """Save TIN mesh, complete mesh, and/or points, with optional cluster suffix."""
    if not (mesh_path or points_path or complete_mesh_path) or not _validate_points_for_export(points):
        return
    
    try:
        from .triangular_mesh_construction import build_tin_mesh, build_complete_pothole_mesh, save_mesh
        
        # Build basic TIN mesh
        mesh = build_tin_mesh(points, engine="auto", z_exaggeration=z_exaggeration)
        
        if mesh_path:
            final_mesh_path = _get_output_path(mesh_path, cluster_id, default_ext=".ply")
            save_mesh(mesh, final_mesh_path)
            print(f"      Saved TIN mesh -> {final_mesh_path}")
        
        if complete_mesh_path:
            try:
                complete_mesh = build_complete_pothole_mesh(
                    points, plane_model,
                    engine="auto", z_exaggeration=z_exaggeration
                )
                final_complete_path = _get_output_path(complete_mesh_path, cluster_id, default_ext=".ply")
                save_mesh(complete_mesh, final_complete_path)
                print(f"      Saved complete pothole mesh -> {final_complete_path}")
            except Exception as e:
                print(f"      Warning: Failed to build complete mesh: {e}")
        
        if points_path:
            verts = np.asarray(mesh.vertices)
            pcd_out = o3d.geometry.PointCloud()
            pcd_out.points = o3d.utility.Vector3dVector(verts)
            final_points_path = _get_output_path(points_path, cluster_id, default_ext=".pcd")
            if o3d.io.write_point_cloud(final_points_path, pcd_out, write_ascii=False, compressed=False):
                print(f"      Saved TIN vertices -> {final_points_path}")
    except Exception:
        pass


def _save_hull_plot(
    points: np.ndarray,
    *,
    hull_path: Optional[str],
    cluster_id: Optional[int] = None,
) -> None:
    """Save 2D convex hull plot with optional cluster suffix."""
    if not hull_path or not _validate_points_for_export(points):
        return
    
    try:
        from .analysis import convex_hull_area
        hull_area, hull_obj = convex_hull_area(points[:, :2])
        if hull_obj is not None:
            final_path = _get_output_path(hull_path, cluster_id, default_ext=".png")
            title = f"Pothole-{cluster_id} (Area={hull_area:.4f} m²)" if cluster_id is not None else f"Hull (Area={hull_area:.4f} m²)"
            save_hull_plot(points[:, :2], hull_obj, final_path, title=title)
            print(f"      Saved 2D hull plot -> {final_path}")
    except Exception:
        pass


def _save_surface_heatmap(
    points: np.ndarray,
    plane_model: np.ndarray,
    *,
    heatmap_path: Optional[str],
    cluster_id: Optional[int] = None,
) -> None:
    """Save surface depth heatmap with optional cluster suffix."""
    if not heatmap_path or not _validate_points_for_export(points):
        return
    
    try:
        coeffs, _ = fit_quadratic_surface(points)
        grid = surface_depth_grid(points, plane_model, coeffs)
        if grid is not None:
            xx, yy, depths_grid, mask = grid
            final_path = _get_output_path(heatmap_path, cluster_id, default_ext=".png")
            save_depth_heatmap(xx, yy, depths_grid, mask, final_path)
            print(f"      Saved surface depth heatmap -> {final_path}")
    except Exception:
        pass


def _save_plane_mesh(
    pcd: o3d.geometry.PointCloud,
    plane_model: np.ndarray,
    extent_points: np.ndarray,
    *,
    plane_path: Optional[str],
    margin: float = 0.1,
) -> None:
    """Save standalone plane mesh sized to extent points."""
    if not plane_path or not _validate_points_for_export(extent_points):
        return
    
    try:
        plane = plane_mesh_covering_cloud(pcd, plane_model, extent_pts=extent_points, margin=margin)
        if plane is not None:
            o3d.io.write_triangle_mesh(plane_path, plane, write_ascii=False)
            print(f"      Saved plane mesh -> {plane_path}")
    except Exception:
        pass


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
    save_delaunay_contrib_2d_path: Optional[str] = None,
    save_delaunay_contrib_3d_multi_path: Optional[str] = None,
    save_tin_mesh_path: Optional[str] = None,
    save_complete_pothole_mesh_path: Optional[str] = None,
    tin_z_exaggeration: float = 1.0,
    save_tin_points_path: Optional[str] = None,
    save_plane_mesh_path: Optional[str] = None,
    plane_extent_margin: float = 0.1,
    save_pothole_points_with_fitted_plane: bool = False,
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
            _save_hull_plot(filtered_points, hull_path=hull_plot_path or "hull_2d.png")
        # Optional surface heatmap for single cluster
        if save_surface_heatmap and 'delaunay_volume' in summary:
            _save_surface_heatmap(filtered_points, plane_model, heatmap_path=surface_heatmap_path or "surface_depth_heatmap.png")
        # Optional Delaunay/TIN contribution visualization (overall)
        _save_delaunay_contrib_images(
            filtered_points, plane_model,
            save_2d_path=save_delaunay_contrib_2d_path,
            save_3d_path=save_delaunay_contrib_3d_multi_path
        )
        # Optional TIN mesh/points export (overall)
        _save_tin_mesh_and_points(
            filtered_points, plane_model,
            mesh_path=save_tin_mesh_path,
            points_path=save_tin_points_path,
            complete_mesh_path=save_complete_pothole_mesh_path,
            z_exaggeration=tin_z_exaggeration
        )
        # Optional standalone plane mesh export (overall)
        _save_plane_mesh(
            pcd, plane_model, filtered_points,
            plane_path=save_plane_mesh_path,
            margin=plane_extent_margin
        )
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
        if 'delaunay_volume' in summary:
            print("    Delaunay-based volume:")
            print(f"      Delaunay volume: {summary['delaunay_volume']:.6f} m³")
            vr = summary.get('volume_ratio_delaunay_over_convex', None)
            if vr is not None:
                print(f"      Volume ratio (surface/convex): {vr:.2f}")

        if save_hull_2d:
            _save_hull_plot(pts, hull_path=hull_plot_path, cluster_id=cluster_id)
        if save_surface_heatmap and 'delaunay_volume' in summary:
            _save_surface_heatmap(pts, plane_model, heatmap_path=surface_heatmap_path, cluster_id=cluster_id)
        # Optional Delaunay/TIN contribution visualization per cluster
        if save_delaunay_contrib_2d_path:
            _save_delaunay_contrib_images(
                pts, plane_model,
                save_2d_path=save_delaunay_contrib_2d_path,
                save_3d_path=None,
                cluster_id=cluster_id
            )
        # Optional TIN mesh/points export per cluster
        _save_tin_mesh_and_points(
            pts, plane_model,
            mesh_path=save_tin_mesh_path,
            points_path=save_tin_points_path,
            complete_mesh_path=save_complete_pothole_mesh_path,
            z_exaggeration=tin_z_exaggeration,
            cluster_id=cluster_id
        )
        if save_pothole_points_with_fitted_plane:
            # Size the plane quad to this cluster's pothole points, with a small margin (meters)
            plane = plane_mesh_covering_cloud(pcd, plane_model, extent_pts=pts, margin=1)
            if plane is not None:
                plane_vertices = np.asarray(plane.vertices)
                plane_triangles = np.asarray(plane.triangles)
                combined_vertices = np.vstack([plane_vertices, pts]) if pts.size > 0 else plane_vertices

                combined_mesh = o3d.geometry.TriangleMesh()
                combined_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
                combined_mesh.triangles = o3d.utility.Vector3iVector(plane_triangles)

                # Build a per-vertex color array:
                # - The first 4 vertices belong to the plane quad → color them blue-ish
                # - Remaining vertices are pothole points → color them red
                colors = np.zeros((combined_vertices.shape[0], 3), dtype=float)
                colors[:4] = np.array([0.2, 0.6, 1.0])
                if combined_vertices.shape[0] > 4:
                    colors[4:] = np.array([1.0, 0.0, 0.0])
                # Attach colors to the mesh so viewers can distinguish plane vs points
                combined_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                # Compute normals for proper shading of the plane triangles
                combined_mesh.compute_vertex_normals()

                pothole_dir = os.path.dirname(pcd_path)
                out_path = os.path.join(pothole_dir, "pothole_points_with_fitted_plane.ply")

                o3d.io.write_triangle_mesh(
                    out_path,
                    combined_mesh,
                    write_ascii=False,
                )
        
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
    filtered_points, depths = filter_pothole_depths(pothole_points, signed_depths)
    if len(filtered_points) == 0:
        result["status"] = "no_points_below_plane"
        return result

    # Single-pothole style summary
    if summary_only and not aggregate_all:
        summary = per_pothole_summary(filtered_points, depths, plane_model, compute_surface=True)
        result["n_clusters"] = 1
        result["clusters"] = [summary]
        result["overall"] = {
            "max_depth": float(depths.max()),
            "mean_depth": float(depths.mean()),
            "median_depth": float(np.median(depths)),
        }
        return result

    labels, n_clusters = dbscan_labels(filtered_points, eps=eps)
    if n_clusters == 0:
        labels, n_clusters = dbscan_labels(filtered_points, eps=EPS_MAX)

    summaries: Dict[int, Dict] = {}
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        pts = filtered_points[cluster_mask]
        dps = depths[cluster_mask]
        summary = per_pothole_summary(pts, dps, plane_model, compute_surface=True)
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
        total_volume = sum(s.get("simple_volume", 0.0) for s in summaries.values())
        avg_of_means = float(np.mean([s.get("mean_depth", 0.0) for s in summaries.values()]))
        result["aggregate"] = {
            "sum_area_hull": float(total_area),
            "sum_volume_hull": float(total_volume),
            "average_of_mean_depths": avg_of_means,
        }

    return result

