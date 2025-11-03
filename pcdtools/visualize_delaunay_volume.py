from __future__ import annotations

"""
Standalone visualizer for Delaunay-based pothole volume contributions using
synthetic generators (from tests) to build a road + pothole scene, fit the
road plane, and visualize per-triangle contributions.

Inputs:
  - PCD/PLY/XYZ with pothole points (Open3D supported)
  - Road plane auto-fit from the generated scene

Outputs:
  - 3D PNG: TIN colored by per-triangle contribution (proxy per-vertex)
  - 2D PNG: XY triangulation colored by per-triangle contribution
  - Prints total Delaunay volume

Usage:
  python visualize_delaunay_volume.py \
    --out3d /abs/path/tin_contrib_3d.png \
    --out2d /abs/path/tin_contrib_2d.png \
    [--seed 42]
"""

import argparse
from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception as _e:  # pragma: no cover
    o3d = None

# Reuse synthetic generators and plane fitting
from test_utilities.test_utils import synthesize_pothole_scene
from pcdtools.plane import segment_plane_road


def z_on_plane(plane: Tuple[float, float, float, float], xy: np.ndarray) -> np.ndarray:
    a, b, c, d = plane
    if abs(c) < 1e-12:
        raise ValueError("Plane nearly vertical; cannot express z(x,y)")
    return -(a * xy[:, 0] + b * xy[:, 1] + d) / c


def delaunay_tris(xy: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import Delaunay  # type: ignore
        tri = Delaunay(xy)
        return np.asarray(tri.simplices, dtype=np.int32)
    except Exception:
        import matplotlib.tri as mtri
        triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
        return np.asarray(triang.triangles, dtype=np.int32)


def triangle_contributions(verts: np.ndarray, tris: np.ndarray, plane: Tuple[float, float, float, float]) -> Tuple[np.ndarray, float]:
    """Return per-triangle volume contributions and their total.

    - verts: (N,3) XYZ vertices of the TIN (pothole points)
    - tris:  (M,3) int indices (Delaunay triangles over XY)
    - plane: (a,b,c,d) road plane, ax+by+cz+d=0

    For each triangle, contribution = area_xy * mean(depths at its 3 vertices),
    where depth = max(z_road(x,y) - z_point, 0).
    """
    # Project vertices to XY and take Z component
    xy = verts[:, :2]
    z = verts[:, 2]
    # Evaluate road plane at each XY to get road height
    z_road = z_on_plane(plane, xy)
    # Vertical separation; clamp to >= 0 so we only integrate below the road
    depth = np.maximum(z_road - z, 0.0)

    contrib = np.empty(len(tris), dtype=float)
    total = 0.0
    for t, (i, j, k) in enumerate(tris):
        # Triangle vertices in XY
        v0, v1, v2 = xy[i], xy[j], xy[k]
        # XY triangle area via 2x2 determinant (0.5 * |(v1-v0) x (v2-v0)|)
        area = 0.5 * abs(np.linalg.det(np.stack([v1 - v0, v2 - v0], axis=0)))
        # Contribution = area * average depth at the 3 vertices
        c = area * (depth[i] + depth[j] + depth[k]) / 3.0
        contrib[t] = c
        total += c
    return contrib, float(total)


def plot_3d_contrib(verts: np.ndarray, tris: np.ndarray, contrib: np.ndarray, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    tri_obj = mtri.Triangulation(verts[:, 0], verts[:, 1], tris)
    # Map per-triangle values to per-vertex by averaging adjacent triangles
    per_vertex = np.zeros(len(verts))
    counts = np.zeros(len(verts))
    for t, (i, j, k) in enumerate(tris):
        for v in (i, j, k):
            per_vertex[v] += contrib[t]
            counts[v] += 1
    per_vertex = np.divide(per_vertex, np.maximum(counts, 1))

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(tri_obj, verts[:, 2], cmap='viridis', linewidth=0.2, antialiased=True, shade=False)
    surf.set_array(per_vertex)
    surf.autoscale()
    fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.02, label='Contribution proxy (m^3)')
    ax.set_title('TIN colored by per-triangle volume contribution')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)


def plot_3d_contrib_multi(
    verts: np.ndarray,
    tris: np.ndarray,
    contrib: np.ndarray,
    out_png: str,
    views: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((25, -60), (30, 30), (60, -120)),
) -> None:
    """Render three 3D views (different elev,azim) of the TIN colored by contribution.

    - verts: (N,3) XYZ vertices
    - tris: (M,3) indices
    - contrib: (M,) per-triangle contributions
    - views: three (elev, azim) tuples
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    tri_obj = mtri.Triangulation(verts[:, 0], verts[:, 1], tris)
    # Per-vertex proxy coloring by averaging adjacent triangle contributions
    per_vertex = np.zeros(len(verts))
    counts = np.zeros(len(verts))
    for t, (i, j, k) in enumerate(tris):
        for v in (i, j, k):
            per_vertex[v] += contrib[t]
            counts[v] += 1
    per_vertex = np.divide(per_vertex, np.maximum(counts, 1))

    vmin = float(np.min(per_vertex))
    vmax = float(np.max(per_vertex)) if np.max(per_vertex) > vmin else vmin + 1.0

    fig = plt.figure(figsize=(15, 5), dpi=120)
    axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]
    mappable = None
    for ax, (elev, azim) in zip(axes, views):
        surf = ax.plot_trisurf(
            tri_obj,
            verts[:, 2],
            cmap='viridis',
            linewidth=0.2,
            antialiased=True,
            shade=False,
            vmin=vmin,
            vmax=vmax,
        )
        surf.set_array(per_vertex)
        surf.autoscale()
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev}, azim={azim}")
        if mappable is None:
            mappable = surf

    cbar = fig.colorbar(mappable, ax=axes, shrink=0.75, pad=0.02, location='right')
    cbar.set_label('Contribution proxy (m^3)')
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)


def plot_2d_contrib(xy: np.ndarray, tris: np.ndarray, contrib: np.ndarray, out_png: str, total: float) -> None:
    # Render a 2D XY triangulation colored by per-triangle volume contribution.
    # Inputs:
    #  - xy: (N,2) triangle vertex coordinates
    #  - tris: (M,3) index triplets into xy
    #  - contrib: (M,) per-triangle contributions (m^3)
    #  - out_png: output path for the PNG
    #  - total: total summed contribution (used in title)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    # Build polygon vertices for each triangle in XY
    polys = [xy[idx] for idx in tris]
    # Normalize contributions to [0,1] using NumPy 2.0-compatible ptp
    rng = float(np.ptp(contrib))
    cmin = float(np.min(contrib))
    scaled = (contrib - cmin) / (rng if rng > 0.0 else 1.0)
    # Map normalized values through a perceptually uniform colormap
    cmap = plt.get_cmap('viridis')
    colors = cmap(scaled)

    # Create figure/axes and draw triangles with light edges
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    pc = PolyCollection(polys, facecolors=colors, edgecolors=(0,0,0,0.2), linewidths=0.5)
    ax.add_collection(pc)
    # Fit limits to content and keep true XY proportions
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    # Add a colorbar keyed to the same colormap/data range
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(contrib)
    fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.02, label='Per-triangle contribution (m^3)')
    # Title with total integrated volume for quick reference
    ax.set_title(f'Delaunay per-triangle contributions (Total={total:.6f} m^3)')
    # Reduce padding and save
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Visualize Delaunay (TIN) volume contributions.')
    ap.add_argument('--seed', type=int, default=42, help='Seed for reproducible synthetic scene')
    ap.add_argument('--out3d', required=True, help='Output PNG for 3D TIN view')
    ap.add_argument('--out2d', required=True, help='Output PNG for 2D XY view')
    args = ap.parse_args()

    # 1) Build a synthetic scene: flat road + one pothole
    scene, labels, meta = synthesize_pothole_scene(
        road_nx=160, road_ny=120, spacing=0.03, road_z=0.0,
        pothole_radius=0.35, pothole_depth=0.06, pothole_points=3500,
        pad_cells=1, seed=args.seed,
    )

    # 2) Fit road plane (RANSAC) on the combined scene
    if o3d is None:  # pragma: no cover
        raise ImportError('open3d is required')
    pcd_scene = o3d.geometry.PointCloud()
    pcd_scene.points = o3d.utility.Vector3dVector(scene)
    plane_model, _ = segment_plane_road(pcd_scene)

    # 3) Use pothole points for TIN (labels: 0=road, 1=pothole)
    verts = np.asarray(scene[labels == 1], dtype=np.float64)
    xy = verts[:, :2]
    tris = delaunay_tris(xy)
    contrib, total = triangle_contributions(verts, tris, tuple(plane_model))

    plot_3d_contrib(verts, tris, contrib, args.out3d)
    plot_2d_contrib(xy, tris, contrib, args.out2d, total)
    print(f"Total Delaunay volume: {total:.6f} m^3")


if __name__ == '__main__':
    main()


