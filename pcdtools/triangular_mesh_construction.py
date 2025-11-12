from __future__ import annotations

"""
TIN (Triangulated Irregular Network) construction and visualization for pothole point clouds.

Reads a .pcd (or other Open3D-supported) point cloud, performs a 2D Delaunay
triangulation over the XY plane, maps triangles to the original 3D Z, builds
an Open3D TriangleMesh, and exports both a mesh file (PLY) and a headless PNG.

Usage (CLI):
  python triangular_mesh_construction.py \
      --pcd /path/to/pothole_points.pcd \
      --out_mesh /path/to/tin.ply \
      --out_png /path/to/tin.png \
      [--only_red] [--red_thr 0.7] [--z_exaggeration 1.0] [--engine auto]

Notes
-----
- If SciPy is available, Delaunay from scipy.spatial is used; otherwise falls
  back to Matplotlib's Triangulation.
- Set --only_red to keep only points colored red (R > red_thr and G,B < 0.3).
"""

import argparse
from typing import Optional, Tuple

import numpy as np
from .strategies import TrianglePruningStrategy
from .strategies.triangle_pruning import _alpha_filter_triangles

try:
    import open3d as o3d
except Exception as _e:  # pragma: no cover
    o3d = None


def _read_pcd(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if o3d is None:  # pragma: no cover
        raise ImportError("open3d is required to read/write point clouds")
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64)
    if cols.size == 0:
        cols = None
    return pts, cols


def _filter_only_red(points: np.ndarray, colors: Optional[np.ndarray], red_thr: float = 0.7) -> np.ndarray:
    if colors is None or colors.shape[0] != points.shape[0]:
        return points
    mask = (colors[:, 0] > float(red_thr)) & (
        colors[:, 1] < 0.3) & (colors[:, 2] < 0.3)
    return points[mask]


def _triangulate_xy(points: np.ndarray, engine: str = "auto") -> np.ndarray:
    """Return integer triangle indices (M,3) for XY Delaunay of points (N,3)."""
    xy = points[:, :2]

    if engine == "scipy" or engine == "auto":
        try:
            from scipy.spatial import Delaunay  # type: ignore
            tri = Delaunay(xy)
            return np.asarray(tri.simplices, dtype=np.int32)
        except Exception:
            if engine == "scipy":
                raise
            # fall through to mpl

    # Matplotlib fallback
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
    return np.asarray(triang.triangles, dtype=np.int32)


def extract_ordered_boundary_loops(faces: np.ndarray) -> list[list[int]]:
    """Extract ordered boundary vertex loops from triangle indices.

    Returns a list of vertex-index lists, each representing a closed boundary ring.
    """
    from collections import defaultdict
    if faces.size == 0:
        return []
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((i, j), (j, k), (k, i)):
            a, b = (u, v) if u < v else (v, u)
            edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1
    boundary_edges = [e for e, c in edge_counts.items() if c == 1]
    if not boundary_edges:
        return []
    neighbors: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    loops: list[list[int]] = []
    visited: set[tuple[int, int]] = set()

    def _mark(u: int, v: int) -> None:
        visited.add((u, v) if u < v else (v, u))

    def _is_visited(u: int, v: int) -> bool:
        key = (u, v) if u < v else (v, u)
        return key in visited

    vertices_with_boundary = sorted(
        set([p for e in boundary_edges for p in e]))
    for start in vertices_with_boundary:
        if all(_is_visited(start, nb) for nb in neighbors[start]):
            continue
        loop: list[int] = []
        prev = -1
        cur = start
        safe_guard = 0
        max_steps = max(8, len(boundary_edges) * 2)
        while safe_guard < max_steps:
            loop.append(cur)
            nbrs = neighbors[cur]
            nxt = None
            for nb in nbrs:
                if nb == prev:
                    continue
                if not _is_visited(cur, nb):
                    nxt = nb
                    break
            if nxt is None and start in nbrs and not _is_visited(cur, start) and start != prev:
                nxt = start
            if nxt is None:
                for nb in nbrs:
                    if nb != prev:
                        nxt = nb
                        break
            if nxt is None:
                break
            _mark(cur, nxt)
            prev, cur = cur, nxt
            safe_guard += 1
            if cur == start:
                break
        if len(loop) >= 3 and loop[0] == cur:
            loops.append(loop)
    return loops


def _point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    """Ray casting point-in-polygon for a single point and polygon in XY.
    poly: (M,2) array of vertices (closed or open)."""
    x, y = float(point[0]), float(point[1])
    inside = False
    n = poly.shape[0]
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        # Check if edge crosses horizontal ray to the right of point
        cond = ((y0 > y) != (y1 > y))
        if cond:
            x_inter = x0 + (y - y0) * (x1 - x0) / (y1 - y0 + 1e-20)
            if x_inter > x:
                inside = not inside
    return inside


def build_plane_with_holes(
    plane_model: np.ndarray,
    boundary_loops: list[list[int]],
    bottom_vertices: np.ndarray,
    extent_points: np.ndarray,
    *,
    margin: float = 0.1,
    grid_res: int = 64,
) -> o3d.geometry.TriangleMesh:
    """Triangulate a road plane mesh with holes defined by the boundary loops.

    - Creates a rectangular grid over the XY extent (with margin)
    - Adds boundary ring vertices (projected to the plane for Z)
    - Triangulates in XY and masks triangles whose centroids fall inside any hole loop
    - Lifts XY to 3D using the plane equation
    """
    import matplotlib.tri as mtri

    if extent_points.shape[0] < 3:
        raise ValueError("Need at least 3 extent points to size plane")

    xy_extent = extent_points[:, :2]
    xmin, ymin = np.min(xy_extent, axis=0) - margin
    xmax, ymax = np.max(xy_extent, axis=0) + margin
    gx = np.linspace(xmin, xmax, int(max(8, grid_res)))
    gy = np.linspace(ymin, ymax, int(max(8, grid_res)))
    XX, YY = np.meshgrid(gx, gy)
    grid_xy = np.stack([XX.ravel(), YY.ravel()], axis=1)

    # Ring vertices in XY from bottom mesh (their XY already lie above the pothole)
    ring_xy_list = []
    for loop in boundary_loops:
        if len(loop) >= 3:
            ring_xy_list.append(
                bottom_vertices[np.asarray(loop, dtype=int), :2])
    if not ring_xy_list:
        # No hole; just build full rectangle plane
        all_xy = grid_xy
        triangles = mtri.Triangulation(all_xy[:, 0], all_xy[:, 1]).triangles
        verts_xy = all_xy
    else:
        ring_xy = np.vstack(ring_xy_list)
        all_xy = np.vstack([grid_xy, ring_xy])
        triang = mtri.Triangulation(all_xy[:, 0], all_xy[:, 1])
        tris = np.asarray(triang.triangles, dtype=np.int32)
        # Mask triangles whose centroid is inside any ring polygon (treat all loops as holes)
        centroids = np.mean(all_xy[tris], axis=1)
        keep = np.ones(tris.shape[0], dtype=bool)
        for loop_xy in ring_xy_list:
            inside = np.array([_point_in_polygon(c, loop_xy)
                              for c in centroids], dtype=bool)
            keep &= ~inside
        triangles = tris[keep]
        verts_xy = all_xy

    # Lift XY to 3D using plane equation
    z = _z_on_plane(plane_model, verts_xy)
    verts_3d = np.column_stack([verts_xy, z])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_3d)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def merge_meshes(mesh_a: o3d.geometry.TriangleMesh, mesh_b: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Concatenate two meshes into one (no vertex deduplication)."""
    va = np.asarray(mesh_a.vertices)
    fa = np.asarray(mesh_a.triangles)
    vb = np.asarray(mesh_b.vertices)
    fb = np.asarray(mesh_b.triangles)
    offset = va.shape[0]
    verts = np.vstack([va, vb]) if vb.size else va.copy()
    tris = np.vstack([fa, fb + offset]) if fb.size else fa.copy()
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(verts)
    out.triangles = o3d.utility.Vector3iVector(tris)
    out.compute_vertex_normals()
    return out


def point_cloud_to_spheres_mesh(
    pcd: o3d.geometry.PointCloud,
    *,
    radius: float = 0.02,
    max_points: int = 5000,
    seed: int = 0,
    resolution: int = 3,
) -> o3d.geometry.TriangleMesh:
    """Approximate a point cloud as a merged mesh of small spheres.

    Args:
        pcd: Open3D point cloud (uses .points and optional .colors)
        radius: sphere radius (units of the input coordinate system)
        max_points: downsample to at most this many points (random)
        seed: RNG seed for reproducible sampling
        resolution: sphere triangulation resolution (3–8 is reasonable)

    Returns:
        TriangleMesh containing one sphere per sampled point
    """
    if len(pcd.points) == 0:
        return o3d.geometry.TriangleMesh()

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if len(
        pcd.colors) == len(pcd.points) else None

    n = pts.shape[0]
    if max_points > 0 and n > max_points:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=int(max_points), replace=False)
        pts = pts[idx]
        if cols is not None and cols.shape[0] == n:
            cols = cols[idx]

    # Base low-poly sphere
    base = o3d.geometry.TriangleMesh.create_sphere(
        radius=float(radius), resolution=int(max(3, resolution)))
    base.compute_vertex_normals()
    vbase = np.asarray(base.vertices)
    fbase = np.asarray(base.triangles)
    vcount = vbase.shape[0]

    all_verts = []
    all_tris = []
    all_cols = []
    offset = 0
    for i, p in enumerate(pts):
        all_verts.append(vbase + p[None, :])
        all_tris.append(fbase + offset)
        if cols is not None:
            c = cols[i]
            all_cols.append(np.tile(c[None, :], (vcount, 1)))
        offset += vcount

    verts = np.vstack(all_verts) if all_verts else np.zeros((0, 3))
    tris = np.vstack(all_tris) if all_tris else np.zeros(
        (0, 3), dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))
    if all_cols:
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(all_cols))
    mesh.compute_vertex_normals()
    return mesh


def build_complete_pothole_mesh(
    pothole_points: np.ndarray,
    plane_model: np.ndarray,
    *,
    engine: str = "auto",
    z_exaggeration: float = 1.0,
    boundary_method: str = "convex_hull"
) -> o3d.geometry.TriangleMesh:
    """
    Build a closed mesh: pothole TIN + vertical walls to road plane.

    Creates a realistic "carved pothole" that connects the pothole surface
    to the road plane via vertical walls around the boundary.

    Args:
        pothole_points: (N,3) pothole surface points
        plane_model: (a,b,c,d) road plane coefficients ax+by+cz+d=0
        engine: triangulation backend
        z_exaggeration: Z scaling factor
        boundary_method: "convex_hull" (only option for now)

    Returns:
        Complete mesh with bottom TIN + side walls + optional top surface
    """
    if pothole_points.shape[0] < 3:
        raise ValueError("Need at least 3 points to build complete mesh")

    # 1. Build bottom TIN (pothole surface)
    bottom_mesh = build_tin_mesh(
        pothole_points, engine=engine, z_exaggeration=z_exaggeration)
    bottom_verts = np.asarray(bottom_mesh.vertices)
    bottom_faces = np.asarray(bottom_mesh.triangles)

    if bottom_faces.size == 0:
        raise ValueError(
            "No triangles in bottom mesh; cannot build complete mesh")

    # 2. Determine boundary vertices from PRUNED mesh state
    #    Prefer extracting boundary loops from triangle edges; fallback to convex hull
    loops = extract_ordered_boundary_loops(bottom_faces)
    if not loops:
        # Fallback to convex hull as a last resort (single ring)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(bottom_verts[:, :2])
            loops = [list(hull.vertices)]
        except Exception:
            raise ValueError(
                "Failed to derive boundary from mesh or convex hull")

    # 3. Project boundary vertices onto road plane and build walls for each loop
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        raise ValueError("Invalid plane normal")
    n = n / n_norm
    d_norm = d / n_norm

    all_verts = bottom_verts.copy()
    all_faces = bottom_faces.copy()
    for loop in loops:
        if len(loop) < 3:
            continue
        ring_idx = np.asarray(loop, dtype=int)
        boundary_3d = bottom_verts[ring_idx]
        dists = boundary_3d @ n + d_norm
        projected = boundary_3d - dists[:, None] * n
        n_bottom = all_verts.shape[0]
        all_verts = np.vstack([all_verts, projected])
        # Build side walls for this loop
        local_faces = []
        for i in range(len(ring_idx)):
            j = (i + 1) % len(ring_idx)
            v0, v1 = int(ring_idx[i]), int(ring_idx[j])
            v2, v3 = n_bottom + i, n_bottom + j
            local_faces.extend([[v0, v2, v1], [v1, v2, v3]])
        all_faces = np.vstack(
            [all_faces, np.asarray(local_faces, dtype=np.int32)])

    # 4. Build final mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(all_verts)
    mesh.triangles = o3d.utility.Vector3iVector(all_faces)
    mesh.compute_vertex_normals()
    return mesh


def build_tin_mesh(points: np.ndarray, *, engine: str = "auto", z_exaggeration: float = 1.0, triangle_pruning: Optional[TrianglePruningStrategy] = None) -> o3d.geometry.TriangleMesh:
    if o3d is None:  # pragma: no cover
        raise ImportError("open3d is required to build meshes")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to build a TIN")

    # Deduplicate XY to avoid zero-area/degenerate triangles
    xy = points[:, :2]
    try:
        u_xy, idx = np.unique(xy, axis=0, return_index=True)
        verts = points[np.sort(idx)].copy()
    except Exception:
        verts = points.copy()

    # Attempt triangulation; if empty, try a tiny jitter to break exact degeneracies
    triangles = _triangulate_xy(verts, engine=engine)
    # Prune flat/skinny simplices
    if triangles.size > 0:
        if triangle_pruning is not None:
            triangles = triangle_pruning.prune(
                verts[:, :2],
                triangles,
                edge_percentile=99.0,      # keep consistent defaults
                circ_percentile=99.0,
                min_circle_ratio=0.02,
            )
        else:
            try:
                import matplotlib.tri as mtri  # type: ignore
                tri_obj = mtri.Triangulation(
                    verts[:, 0], verts[:, 1], triangles)
                analyzer = mtri.TriAnalyzer(tri_obj)
                flat_mask = analyzer.get_flat_tri_mask(min_circle_ratio=0.02)
                triangles = triangles[~flat_mask]
            except Exception:
                pass
            triangles = _alpha_filter_triangles(
                verts[:, :2], triangles,
                edge_percentile=99.0,      # very permissive
                circ_percentile=99.0       # very permissive
            )
    if triangles.size == 0:
        # Jitter scale based on data extent (very small relative noise)
        span = float(
            np.max(np.ptp(verts[:, :2], axis=0))) if verts.shape[0] >= 2 else 1.0
        jitter = max(span, 1.0) * 1e-9
        rng = np.random.default_rng(0)
        verts_j = verts.copy()
        verts_j[:, 0:2] += rng.normal(0.0, jitter, size=(verts.shape[0], 2))
        tri2 = _triangulate_xy(verts_j, engine=engine)
        if tri2.size > 0:
            # Apply pruning again after jitter
            if triangle_pruning is not None:
                tri2 = triangle_pruning.prune(
                    verts_j[:, :2],
                    tri2,
                    edge_percentile=99.0,
                    circ_percentile=99.0,
                    min_circle_ratio=0.02,
                )
            else:
                try:
                    import matplotlib.tri as mtri  # type: ignore
                    tri_obj2 = mtri.Triangulation(
                        verts_j[:, 0], verts_j[:, 1], tri2)
                    analyzer2 = mtri.TriAnalyzer(tri_obj2)
                    flat_mask2 = analyzer2.get_flat_tri_mask(
                        min_circle_ratio=0.02)
                    tri2 = tri2[~flat_mask2]
                except Exception:
                    pass
                tri2 = _alpha_filter_triangles(
                    verts_j[:, :2], tri2,
                    edge_percentile=99.0,
                    circ_percentile=99.0
                )
            verts = verts_j
            triangles = tri2

    if z_exaggeration != 1.0:
        verts[:, 2] *= float(z_exaggeration)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def save_mesh(mesh: o3d.geometry.TriangleMesh, out_mesh: str) -> None:
    ok = o3d.io.write_triangle_mesh(out_mesh, mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to save mesh -> {out_mesh}")


def save_tin_png(points: np.ndarray, triangles: np.ndarray, out_png: str, *, title: str = "TIN") -> None:
    """Headless PNG render via Matplotlib plot_trisurf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    triang = mtri.Triangulation(x, y, triangles)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(triang, z, cmap="viridis",
                           linewidth=0.2, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.02, label="Z")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Nice view
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    import numpy as np


def _z_on_plane(plane, xy):
    a, b, c, d = plane
    if abs(c) < 1e-12:
        raise ValueError("Plane nearly vertical; cannot express z(x,y).")
    return -(a*xy[:, 0] + b*xy[:, 1] + d) / c


def volume_under_road_plane_from_tin(vertices, triangles, road_plane, clamp_positive=True):
    """
    vertices: (N,3) TIN vertices (from your pothole mesh)
    triangles: (M,3) int indices into vertices
    road_plane: (a,b,c,d) for ax+by+cz+d=0
    Returns volume in same units^3 as your inputs (e.g., m^3).
    """
    xy = vertices[:, :2]
    z = vertices[:, 2]
    z_road = _z_on_plane(road_plane, xy)
    depth = z_road - z  # positive if vertex is below the road plane

    V = 0.0
    for i, j, k in triangles:
        v0, v1, v2 = xy[i], xy[j], xy[k]
        area = 0.5 * abs(np.linalg.det(np.stack([v1 - v0, v2 - v0], axis=0)))
        d0, d1, d2 = depth[i], depth[j], depth[k]
        if clamp_positive:
            d0 = max(d0, 0.0)
            d1 = max(d1, 0.0)
            d2 = max(d2, 0.0)
        V += area * (d0 + d1 + d2) / 3.0
    return float(V)


def main():
    ap = argparse.ArgumentParser(
        description="Build a TIN from a pothole point cloud and visualize it.")
    ap.add_argument("--pcd", required=True,
                    help="Input PCD/PLY/XYZ path (pothole points)")
    ap.add_argument("--out_mesh", required=True,
                    help="Output mesh path (.ply recommended)")
    ap.add_argument("--out_png", required=True,
                    help="Output PNG visualization path")
    ap.add_argument("--engine", choices=["auto", "scipy", "mpl"],
                    default="auto", help="Triangulation backend")
    ap.add_argument("--only_red", action="store_true",
                    help="Keep only red points (R>thr, G,B<0.3)")
    ap.add_argument("--red_thr", type=float, default=0.7,
                    help="Threshold for red channel when --only_red is set")
    ap.add_argument("--z_exaggeration", type=float, default=1.0,
                    help="Multiply Z for visualization/mesh (>=0)")
    args = ap.parse_args()

    pts, cols = _read_pcd(args.pcd)
    if args.only_red:
        pts = _filter_only_red(pts, cols, red_thr=args.red_thr)
        if pts.shape[0] < 3:
            raise ValueError(
                "Not enough red points to triangulate (need >= 3)")

    mesh = build_tin_mesh(pts, engine=args.engine,
                          z_exaggeration=args.z_exaggeration)
    save_mesh(mesh, args.out_mesh)
    # Reuse triangles/vertices from mesh for PNG
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    save_tin_png(verts, tris, args.out_png,
                 title="TIN (XY Delaunay, Z as height)")

    # V = volume_under_road_plane_from_tin(verts, tris, road_plane)


if __name__ == "__main__":
    main()
