from __future__ import annotations

import numpy as np


def save_hull_plot(points_xy: np.ndarray, hull, out_path: str, title: str = "Hull"):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(points_xy[:, 0], points_xy[:, 1], s=4, c='k', alpha=0.4, label='points')
        hv = hull.vertices
        cycle = list(hv) + [hv[0]]
        ax.plot(points_xy[cycle, 0], points_xy[cycle, 1], 'r-', lw=2, label='convex hull')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass



def save_depth_heatmap(
    xx: np.ndarray,
    yy: np.ndarray,
    depths_grid: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    title: str = "Surface depth heatmap (m)",
) -> None:
    """Save a heatmap of the surface-based depths. Areas outside the pothole mask are hidden."""
    try:
        import matplotlib.pyplot as plt
        depths = np.array(depths_grid, dtype=float)
        masked = np.ma.masked_where(~mask, depths)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            masked,
            origin='lower',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap='viridis'
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label('Depth (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def plane_mesh_covering_cloud(pcd, plane_model, extent_pts=None, margin: float = 0.1, color=(0.2, 0.6, 1.0)):
    """Create an Open3D mesh for the fitted plane sized to points of interest.

    - If extent_pts is provided (Nx3), the plane quad bounds only those points
      (plus a small margin in meters). Otherwise, it falls back to the cloud AABB.
    - Returns a 2-triangle quad (first 4 vertices) colored uniformly.
    """
    try:
        import open3d as o3d
        import numpy as np
    except Exception:
        return None

    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=float)
    n_norm = np.linalg.norm(n) + 1e-12
    n = n / n_norm
    d = d / n_norm
    # !! Normalize plane so normal has unit length; keeps extents and distances consistent

    # Build orthonormal basis {n, u, v}
    a_vec = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a_vec)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    # !! Create two in-plane unit vectors (u, v) orthogonal to the normal n

    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return None

    # Choose centroid source
    if extent_pts is not None and hasattr(extent_pts, "size") and np.asarray(extent_pts).size > 0:
        extent_pts_np = np.asarray(extent_pts, dtype=float)
        centroid_world = extent_pts_np.mean(axis=0)
    else:
        centroid_world = pts.mean(axis=0)
    # !! Center the quad around the centroid of the selected points (or entire cloud)

    # Project centroid onto plane to get the quad center
    t = (np.dot(n, centroid_world) + d)
    center = centroid_world - t * n
    # !! Move the centroid onto the plane to serve as the quad center

    if extent_pts is not None and hasattr(extent_pts, "size") and np.asarray(extent_pts).size > 0:
        # Compute extents in plane-local coordinates for the provided points
        rel = np.asarray(extent_pts, dtype=float) - center
        u_coords = rel @ u
        v_coords = rel @ v
        umin, umax = float(u_coords.min() - margin), float(u_coords.max() + margin)
        vmin, vmax = float(v_coords.min() - margin), float(v_coords.max() + margin)
        # !! Project points into (u,v) plane coordinates and expand bounds by margin (meters)
        corners_local = np.array([
            [umin, vmin],
            [umax, vmin],
            [umax, vmax],
            [umin, vmax],
        ], dtype=float)
        v0 = center + corners_local[0, 0] * u + corners_local[0, 1] * v
        v1 = center + corners_local[1, 0] * u + corners_local[1, 1] * v
        v2 = center + corners_local[2, 0] * u + corners_local[2, 1] * v
        v3 = center + corners_local[3, 0] * u + corners_local[3, 1] * v
    else:
        # Fallback: cover the whole cloud AABB with a small scale factor
        aabb = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(aabb.get_extent())
        half = 0.5 * diag * 1.05
        # !! No specific points given: use overall cloud size as a conservative plane extent
        v0 = center + (-half) * u + (-half) * v
        v1 = center + (half) * u + (-half) * v
        v2 = center + (half) * u + (half) * v
        v3 = center + (-half) * u + (half) * v

    verts = o3d.utility.Vector3dVector(np.stack([v0, v1, v2, v3], axis=0))
    tris = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    mesh = o3d.geometry.TriangleMesh(vertices=verts, triangles=tris)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    # !! First 4 vertices define a quad; two triangles render the plane surface
    return mesh


def visualize_plane_and_potholes(
    pcd,
    plane_model,
    pothole_points: np.ndarray,
    labels: np.ndarray | None = None,
    n_clusters: int | None = None,
) -> None:
    """Open an Open3D viewer with the plane mesh and pothole points (colored by cluster)."""
    try:
        import open3d as o3d
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception:
        return

    mesh = plane_mesh_covering_cloud(pcd, plane_model)
    geoms = []
    if mesh is not None:
       geoms.append(mesh)

    pothole_pcd = o3d.geometry.PointCloud()
    pothole_pcd.points = o3d.utility.Vector3dVector(pothole_points)

    colors = np.zeros((len(pothole_points), 3))
    if labels is not None and n_clusters is not None and len(pothole_points) > 0:
        cmap = plt.get_cmap('tab10')
        for cid in range(n_clusters):
            mask = labels == cid
            colors[mask] = cmap(cid % 10)[:3]
        colors[labels == -1] = [0.5, 0.5, 0.5]
    else:
        colors[:] = [1.0, 0.0, 0.0]

    pothole_pcd.colors = o3d.utility.Vector3dVector(colors)
    geoms.append(pothole_pcd)

    try:
        o3d.visualization.draw_geometries(
            geoms,
            window_name="Plane + Pothole Points",
            width=1200,
            height=800,
            left=50,
            top=50,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=True,
        )
    except Exception:
        pass

