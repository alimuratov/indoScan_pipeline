"""Artifact export functions for pipeline outputs.

Handles saving meshes, plots, heatmaps, and other visualization artifacts.
Separates I/O concerns from core pipeline logic.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING
import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from .strategies import TrianglePruningStrategy
    from .pipeline import VisualizationConfig, OutputPathsConfig


def build_pruned_delaunay_triangles(
    points: np.ndarray,
    *,
    edge_percentile: float = 99.0,
    circ_percentile: float = 99.0,
    min_circle_ratio: float = 0.005,
    pruning: Optional["TrianglePruningStrategy"] = None,
) -> np.ndarray:
    """Build Delaunay triangles with flatness and alpha pruning (consistent with TIN builder)."""
    if points.shape[0] < 3:
        return np.array([], dtype=np.int32).reshape(0, 3)

    try:
        from .visualize_delaunay_volume import delaunay_tris
        from .strategies import PerimeterOnlyPruningStrategy

        xy = points[:, :2]
        tris = delaunay_tris(xy)

        if tris.size > 0:
            strategy = pruning or PerimeterOnlyPruningStrategy()
            tris = strategy.prune(
                xy,
                tris,
                edge_percentile=edge_percentile,
                circ_percentile=circ_percentile,
                min_circle_ratio=min_circle_ratio,
            )

        return tris
    except Exception:
        return np.array([], dtype=np.int32).reshape(0, 3)


def get_output_path(base_path: str, cluster_id: Optional[int], default_ext: str = ".png") -> str:
    """Generate output path with optional cluster suffix."""
    if cluster_id is None:
        return base_path
    base, ext = os.path.splitext(base_path)
    return f"{base}_cluster_{cluster_id}{ext or default_ext}"


def validate_points_for_export(points: np.ndarray, min_points: int = 3) -> bool:
    """Common validation for point-based exports."""
    return points.ndim == 2 and points.shape[1] == 3 and points.shape[0] >= min_points


def save_delaunay_contrib_images(
    points: np.ndarray,
    plane_model: np.ndarray,
    *,
    save_2d_path: Optional[str] = None,
    save_3d_path: Optional[str] = None,
    cluster_id: Optional[int] = None,
    pruning: Optional["TrianglePruningStrategy"] = None,
) -> None:
    """Save Delaunay contribution visualizations using pruned triangles."""
    if not (save_2d_path or save_3d_path) or points.shape[0] < 3:
        return

    try:
        from .visualize_delaunay_volume import triangle_contributions, plot_2d_contrib, plot_3d_contrib_multi

        tris = build_pruned_delaunay_triangles(points, pruning=pruning)
        if tris.size == 0:
            return

        xy = points[:, :2]
        contrib, total = triangle_contributions(
            points, tris, tuple(plane_model))

        if save_2d_path:
            final_2d_path = get_output_path(
                save_2d_path, cluster_id, default_ext=".png")
            plot_2d_contrib(xy, tris, contrib, final_2d_path, total)
            print(f"      Saved Delaunay 2D contrib -> {final_2d_path}")
        if save_3d_path:
            final_3d_path = get_output_path(
                save_3d_path, cluster_id, default_ext=".png")
            plot_3d_contrib_multi(points, tris, contrib, final_3d_path)
            print(f"      Saved Delaunay 3-view 3D contrib -> {final_3d_path}")
    except Exception:
        pass


def save_tin_mesh_and_points(
    points: np.ndarray,
    plane_model: np.ndarray,
    *,
    viz_cfg: "VisualizationConfig",
    out_cfg: "OutputPathsConfig",
    pcd: o3d.geometry.PointCloud,
    plane_mesh_margin: float,
    plane_with_hole_grid_res: int,
    cluster_id: Optional[int] = None,
    triangle_pruning: Optional["TrianglePruningStrategy"] = None,
) -> None:
    """Save TIN mesh, complete mesh, plane-with-hole, combined mesh, and/or points.

    All paths may include a cluster suffix when cluster_id is provided.
    """
    if not (out_cfg.save_tin_mesh_path or out_cfg.save_tin_points_path or
            out_cfg.save_complete_pothole_mesh_path or out_cfg.save_plane_with_hole_mesh_path or
            out_cfg.save_combined_mesh_path or out_cfg.save_overlay_with_pcd_mesh_path) or not validate_points_for_export(points):
        return

    try:
        from .triangular_mesh_construction import (
            build_tin_mesh,
            build_complete_pothole_mesh,
            save_mesh,
            extract_ordered_boundary_loops,
            build_plane_with_holes,
            merge_meshes,
            point_cloud_to_spheres_mesh,
        )

        # Build basic TIN mesh
        mesh = build_tin_mesh(points, engine="auto",
                              z_exaggeration=viz_cfg.tin_z_exaggeration,
                              triangle_pruning=triangle_pruning)

        if out_cfg.save_tin_mesh_path:
            final_mesh_path = get_output_path(
                out_cfg.save_tin_mesh_path, cluster_id, default_ext=".ply")
            save_mesh(mesh, final_mesh_path)
            print(f"      Saved TIN mesh -> {final_mesh_path}")

        # Prepare boundary loops from pruned TIN
        bottom_verts = np.asarray(mesh.vertices)
        bottom_faces = np.asarray(mesh.triangles)
        loops = extract_ordered_boundary_loops(bottom_faces)

        plane_with_hole_mesh = None
        if out_cfg.save_plane_with_hole_mesh_path or out_cfg.save_combined_mesh_path:
            if loops:
                plane_with_hole_mesh = build_plane_with_holes(
                    plane_model,
                    loops,
                    bottom_verts,
                    points,
                    margin=plane_mesh_margin,
                    grid_res=int(max(8, plane_with_hole_grid_res)),
                )
                if out_cfg.save_plane_with_hole_mesh_path:
                    final_pwh_path = get_output_path(
                        out_cfg.save_plane_with_hole_mesh_path, cluster_id, default_ext=".ply")
                    save_mesh(plane_with_hole_mesh, final_pwh_path)
                    print(
                        f"      Saved plane-with-hole mesh -> {final_pwh_path}")
            else:
                print("      Warning: No boundary loops; plane-with-hole not created")

        if out_cfg.save_complete_pothole_mesh_path:
            try:
                complete_mesh = build_complete_pothole_mesh(
                    points, plane_model,
                    engine="auto", z_exaggeration=viz_cfg.tin_z_exaggeration
                )
                final_complete_path = get_output_path(
                    out_cfg.save_complete_pothole_mesh_path, cluster_id, default_ext=".ply")
                save_mesh(complete_mesh, final_complete_path)
                print(
                    f"      Saved complete pothole mesh -> {final_complete_path}")
            except Exception as e:
                print(f"      Warning: Failed to build complete mesh: {e}")
        else:
            complete_mesh = None

        # Optionally save a combined mesh (complete pothole + plane-with-hole) and/or overlay with input PCD spheres
        if out_cfg.save_combined_mesh_path or out_cfg.save_overlay_with_pcd_mesh_path:
            try:
                if complete_mesh is None:
                    # Build complete mesh on the fly for combination
                    complete_mesh = build_complete_pothole_mesh(
                        points, plane_model,
                        engine="auto", z_exaggeration=viz_cfg.tin_z_exaggeration
                    )
                if plane_with_hole_mesh is None:
                    if loops:
                        plane_with_hole_mesh = build_plane_with_holes(
                            plane_model,
                            loops,
                            bottom_verts,
                            points,
                            margin=plane_mesh_margin,
                            grid_res=int(max(8, plane_with_hole_grid_res)),
                        )
                    else:
                        plane_with_hole_mesh = None
                merged = None
                if plane_with_hole_mesh is not None and complete_mesh is not None:
                    merged = merge_meshes(complete_mesh, plane_with_hole_mesh)
                elif complete_mesh is not None:
                    merged = complete_mesh
                elif plane_with_hole_mesh is not None:
                    merged = plane_with_hole_mesh
                # Save combined (without PCD) if requested
                if out_cfg.save_combined_mesh_path and merged is not None:
                    final_combined_path = get_output_path(
                        out_cfg.save_combined_mesh_path, cluster_id, default_ext=".ply")
                    save_mesh(merged, final_combined_path)
                    print(
                        f"      Saved combined mesh -> {final_combined_path}")

                # Save overlay (combined + input PCD spheres) if requested
                if out_cfg.save_overlay_with_pcd_mesh_path and pcd is not None:
                    try:
                        pcd_mesh = point_cloud_to_spheres_mesh(
                            pcd,
                            radius=viz_cfg.pcd_sphere_radius,
                            max_points=int(max(0, viz_cfg.pcd_max_points)),
                            seed=int(viz_cfg.pcd_random_seed),
                            resolution=int(
                                max(3, viz_cfg.pcd_sphere_resolution)),
                        )
                        overlay_merged = pcd_mesh if merged is None else merge_meshes(
                            merged, pcd_mesh)
                        final_overlay_path = get_output_path(
                            out_cfg.save_overlay_with_pcd_mesh_path, cluster_id, default_ext=".ply")
                        save_mesh(overlay_merged, final_overlay_path)
                        print(
                            f"      Saved overlay (mesh + PCD) -> {final_overlay_path}")
                    except Exception as e:
                        print(
                            f"      Warning: Failed to build overlay mesh: {e}")
            except Exception as e:
                print(f"      Warning: Failed to build combined mesh: {e}")

        if out_cfg.save_tin_points_path:
            verts = np.asarray(mesh.vertices)
            pcd_out = o3d.geometry.PointCloud()
            pcd_out.points = o3d.utility.Vector3dVector(verts)
            final_points_path = get_output_path(
                out_cfg.save_tin_points_path, cluster_id, default_ext=".pcd")
            if o3d.io.write_point_cloud(final_points_path, pcd_out, write_ascii=False, compressed=False):
                print(f"      Saved TIN vertices -> {final_points_path}")
    except Exception:
        pass


def save_hull_plot(
    points: np.ndarray,
    *,
    hull_path: Optional[str],
    cluster_id: Optional[int] = None,
) -> None:
    """Save 2D convex hull plot with optional cluster suffix."""
    if not hull_path or not validate_points_for_export(points):
        return

    try:
        from .analysis import convex_hull_area
        from .visualize import save_hull_plot as save_hull_plot_impl

        hull_area, hull_obj = convex_hull_area(points[:, :2])
        if hull_obj is not None:
            final_path = get_output_path(
                hull_path, cluster_id, default_ext=".png")
            title = f"Pothole-{cluster_id} (Area={hull_area:.4f} m²)" if cluster_id is not None else f"Hull (Area={hull_area:.4f} m²)"
            save_hull_plot_impl(
                points[:, :2], hull_obj, final_path, title=title)
            print(f"      Saved 2D hull plot -> {final_path}")
    except Exception:
        pass


def save_surface_heatmap(
    points: np.ndarray,
    plane_model: np.ndarray,
    *,
    heatmap_path: Optional[str],
    cluster_id: Optional[int] = None,
) -> None:
    """Save surface depth heatmap with optional cluster suffix."""
    if not heatmap_path or not validate_points_for_export(points):
        return

    try:
        from .analysis import fit_quadratic_surface, surface_depth_grid
        from .visualize import save_depth_heatmap

        coeffs, _ = fit_quadratic_surface(points)
        grid = surface_depth_grid(points, plane_model, coeffs)
        if grid is not None:
            xx, yy, depths_grid, mask = grid
            final_path = get_output_path(
                heatmap_path, cluster_id, default_ext=".png")
            save_depth_heatmap(xx, yy, depths_grid, mask, final_path)
            print(f"      Saved surface depth heatmap -> {final_path}")
    except Exception:
        pass


def save_plane_mesh(
    pcd: o3d.geometry.PointCloud,
    plane_model: np.ndarray,
    extent_points: np.ndarray,
    *,
    plane_path: Optional[str],
    margin: float = 0.1,
) -> None:
    """Save standalone plane mesh sized to extent points."""
    if not plane_path or not validate_points_for_export(extent_points):
        return

    try:
        from .visualize import plane_mesh_covering_cloud

        plane = plane_mesh_covering_cloud(
            pcd, plane_model, extent_pts=extent_points, margin=margin)
        if plane is not None:
            o3d.io.write_triangle_mesh(plane_path, plane, write_ascii=False)
            print(f"      Saved plane mesh -> {plane_path}")
    except Exception:
        pass


def save_points_with_fitted_plane_mesh(
    pcd: o3d.geometry.PointCloud,
    plane_model: np.ndarray,
    pts: np.ndarray,
    pcd_path: str,
) -> None:
    """Save a helper mesh containing the fitted plane quad and pothole points."""
    try:
        from .visualize import plane_mesh_covering_cloud

        plane = plane_mesh_covering_cloud(
            pcd, plane_model, extent_pts=pts, margin=1)
        if plane is None:
            return
        plane_vertices = np.asarray(plane.vertices)
        plane_triangles = np.asarray(plane.triangles)
        combined_vertices = np.vstack(
            [plane_vertices, pts]) if pts.size > 0 else plane_vertices

        combined_mesh = o3d.geometry.TriangleMesh()
        combined_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
        combined_mesh.triangles = o3d.utility.Vector3iVector(plane_triangles)

        colors = np.zeros((combined_vertices.shape[0], 3), dtype=float)
        colors[:4] = np.array([0.2, 0.6, 1.0])
        if combined_vertices.shape[0] > 4:
            colors[4:] = np.array([1.0, 0.0, 0.0])
        combined_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        combined_mesh.compute_vertex_normals()

        pothole_dir = os.path.dirname(pcd_path)
        out_path = os.path.join(
            pothole_dir, "pothole_points_with_fitted_plane.ply")
        o3d.io.write_triangle_mesh(out_path, combined_mesh, write_ascii=False)
    except Exception:
        pass


def visualize_input_and_mesh(
    pcd: o3d.geometry.PointCloud,
    mesh_path: Optional[str],
) -> None:
    """Open an interactive viewer overlaying the input PCD with a mesh path."""
    if not mesh_path:
        return
    try:
        if not os.path.exists(mesh_path):
            print(
                f"      Warning: Mesh path not found for overlay -> {mesh_path}")
            return
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh.is_empty():
            print(f"      Warning: Mesh empty for overlay -> {mesh_path}")
            return
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([pcd, mesh])
    except Exception:
        pass


def export_artifacts_for_cluster(
    pts: np.ndarray,
    plane_model: np.ndarray,
    *,
    cluster_id: Optional[int],
    has_surface: bool,
    viz_cfg: "VisualizationConfig",
    out_cfg: "OutputPathsConfig",
    pcd: o3d.geometry.PointCloud,
    plane_extent_margin: float,
    plane_with_hole_grid_res: int,
    pcd_path: str,
    triangle_pruning: Optional["TrianglePruningStrategy"] = None,
) -> None:
    """Shared artifact export for single or per-cluster processing."""
    # Hull plot
    if viz_cfg.save_hull_2d:
        save_hull_plot(pts, hull_path=out_cfg.hull_plot_path,
                       cluster_id=cluster_id)

    # Heatmap (only when we have surface/Delaunay context)
    if viz_cfg.save_surface_heatmap and has_surface:
        save_surface_heatmap(
            pts,
            plane_model,
            heatmap_path=out_cfg.surface_heatmap_path,
            cluster_id=cluster_id,
        )

    # Delaunay contribution images (3D only for overall/single cluster)
    save_3d = out_cfg.save_delaunay_contrib_3d_multi_path if cluster_id is None else None
    if out_cfg.save_delaunay_contrib_2d_path or save_3d:
        save_delaunay_contrib_images(
            pts,
            plane_model,
            save_2d_path=out_cfg.save_delaunay_contrib_2d_path,
            save_3d_path=save_3d,
            cluster_id=cluster_id,
            pruning=triangle_pruning,
        )

    # TIN meshes/points/combined/overlay
    save_tin_mesh_and_points(
        pts,
        plane_model,
        viz_cfg=viz_cfg,
        out_cfg=out_cfg,
        pcd=pcd,
        plane_mesh_margin=plane_extent_margin,
        plane_with_hole_grid_res=plane_with_hole_grid_res,
        cluster_id=cluster_id,
        triangle_pruning=triangle_pruning,
    )

    # Standalone plane mesh + overlay viewer only for overall (single-pothole path)
    if cluster_id is None:
        if out_cfg.save_plane_mesh_path:
            save_plane_mesh(
                pcd,
                plane_model,
                pts,
                plane_path=out_cfg.save_plane_mesh_path,
                margin=plane_extent_margin,
            )
        if viz_cfg.visualize_overlay:
            target_mesh = out_cfg.save_combined_mesh_path or out_cfg.save_complete_pothole_mesh_path
            visualize_input_and_mesh(pcd, target_mesh)

    # Helper mesh with fitted plane and pothole points (per-cluster)
    if viz_cfg.save_pothole_points_with_fitted_plane:
        save_points_with_fitted_plane_mesh(pcd, plane_model, pts, pcd_path)
