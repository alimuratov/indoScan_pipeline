# scripts/pcdtools/synthetic.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scripts.pcdtools.robust_pothole_filter import filter_pothole_candidates

try:
    import open3d as o3d  # optional
except Exception:  # pragma: no cover
    o3d = None


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GridMeta:
    """Metadata describing a rectangular road grid."""
    nx: int
    ny: int
    spacing: float
    origin: Tuple[float, float] = (0.0, 0.0)  # (x0, y0)

    @property
    def x0(self) -> float:
        return self.origin[0]

    @property
    def y0(self) -> float:
        return self.origin[1]

    @property
    def x1(self) -> float:
        return self.x0 + (self.nx - 1) * self.spacing

    @property
    def y1(self) -> float:
        return self.y0 + (self.ny - 1) * self.spacing


def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def _to_o3d(points: np.ndarray, colors: Optional[np.ndarray] = None):
    """Convert to Open3D PointCloud if available; otherwise raise a clear error."""
    if o3d is None:
        raise ImportError(
            "open3d is not installed. Install `open3d` or use the numpy arrays directly."
        )
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(float, copy=False))
    if colors is not None and colors.shape == (points.shape[0], 3):
        pc.colors = o3d.utility.Vector3dVector(colors.astype(float, copy=False))
    return pc


# ---------------------------------------------------------------------
# 1) Generate a flat road surface (grid with optional jitter/roughness)
# ---------------------------------------------------------------------

def generate_flat_road_surface(
    nx: int = 300,
    ny: int = 200,
    spacing: float = 0.02,
    z: float = 0.0,
    jitter_xy: float = 0.0,
    jitter_z: float = 0.0,
    origin: Tuple[float, float] = (0.0, 0.0),
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, GridMeta]:
    """
    Create a flat (or slightly rough) road surface as a regular XY grid.

    Returns
    -------
    points : (N, 3) float64
        Road surface point cloud, row-major ordering (y as rows, x as columns).
    meta : GridMeta
        Grid metadata needed for carving (deleting road points above potholes).

    Notes
    -----
    - `jitter_xy` adds small random XY wobble to each grid point (meters).
    - `jitter_z` adds small random Z noise to the road height.
    """
    g = _rng(seed)
    x = np.arange(nx, dtype=float) * spacing + origin[0]
    y = np.arange(ny, dtype=float) * spacing + origin[1]
    xx, yy = np.meshgrid(x, y)  # shape (ny, nx)

    zz = np.full_like(xx, float(z))
    if jitter_z > 0:
        zz = zz + g.normal(0.0, jitter_z, size=zz.shape)

    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    if jitter_xy > 0:
        pts[:, 0:2] += g.normal(0.0, jitter_xy, size=(pts.shape[0], 2))

    meta = GridMeta(nx=nx, ny=ny, spacing=spacing, origin=origin)
    return pts.astype(np.float64, copy=False), meta


# ---------------------------------------------------------------------
# 2–3) Generate a bell‑shaped pothole near the road surface
# ---------------------------------------------------------------------

def generate_bell_pothole_points(
    center: Tuple[float, float] = (2.0, 2.0),
    radius: float = 0.25,
    depth: float = 0.05,
    n_points: int = 4000,
    sigma: Optional[float] = None,
    xy_jitter: float = 0.003,
    z_noise: float = 0.002,
    elliptical: float = 0.25,
    tilt: float = 0.0,
    seed: Optional[int] = None,
    return_colors: bool = False,
) -> np.ndarray:
    """
    Sample a bowl-shaped (bell/gaussian) pothole just below the road surface (z <= 0).

    The bowl profile is:
        z = -depth * exp(-0.5 * r_e^2 / sigma^2) + noise
    where r_e is an elliptical radius (to add realism).

    Parameters
    ----------
    center : (cx, cy)
        XY center of the pothole (meters).
    radius : float
        Max XY extent to sample (disk radius). Sigma defaults to radius/2 if not given.
    depth : float
        Maximum depth at the center (meters, positive number => depression).
    n_points : int
        Number of points to sample in the pothole.
    sigma : Optional[float]
        Spread of the Gaussian. If None, uses radius/2.
    xy_jitter : float
        Extra XY jitter to roughen edges (meters).
    z_noise : float
        Additive Gaussian noise in Z (meters).
    elliptical : float
        Degree of ellipticity (0 => perfectly circular). Typical 0–0.4.
    tilt : float
        Optional tilt (slope) applied to the bowl along x (meters per meter).
    seed : Optional[int]
        RNG seed for reproducibility.
    color : Optional[Tuple[int, int, int]]
        Color of the pothole points.
    Returns
    -------
    pothole_points : (n_points, 3) float64
        Points produce a realistic, noisy depression centered at `center`.
    """
    g = _rng(seed)
    sigma = radius / 2.0 if sigma is None else float(sigma)

    # Randomly sample a disk of radius `radius` (area-uniform)
    u = g.random(n_points)
    r = radius * np.sqrt(u)
    theta = 2.0 * math.pi * g.random(n_points)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)

    if xy_jitter > 0:
        x += g.normal(0.0, xy_jitter, size=n_points)
        y += g.normal(0.0, xy_jitter, size=n_points)

    # Elliptical deformation (stretch axes a,b)
    if elliptical > 0:
        a = 1.0 + elliptical
        b = 1.0 - elliptical
        # Compute elliptical radius^2
        dx = (x - center[0]) / a
        dy = (y - center[1]) / b
        r2 = dx * dx + dy * dy
    else:
        dx = x - center[0]
        dy = y - center[1]
        r2 = dx * dx + dy * dy

    # Bell / Gaussian bowl (depression => negative z relative to road plane z=0)
    z = -depth * np.exp(-0.5 * r2 / (sigma * sigma))

    # Apply a slight tilt (optional) to break symmetry further
    if tilt != 0.0:
        z += tilt * (x - center[0])

    # Add surface roughness
    if z_noise > 0:
        z += g.normal(0.0, z_noise, size=n_points)

    pts = np.column_stack([x, y, z]).astype(np.float64, copy=False)
    if not return_colors:
        return pts
    # Produce solid red colors for pothole visualization
    cols = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (n_points, 1))
    return pts, cols


# ---------------------------------------------------------------------
# 4) Delete road surface points directly above pothole points
# ---------------------------------------------------------------------

def delete_road_points_above_pothole(
    road_points: np.ndarray,
    road_meta: GridMeta,
    pothole_points: np.ndarray,
    pad_cells: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove road points whose XY lies directly above the pothole points.

    For efficiency and SciPy-free operation, we project both clouds onto the
    *road grid* using rounding to the nearest cell (based on `GridMeta.spacing`),
    then drop any road cell touched by at least one pothole point. Optionally
    dilate the deletion mask by `pad_cells` to widen the hole rim.

    Parameters
    ----------
    road_points : (Nr,3)
        Road surface points created by `generate_flat_road_surface`.
    road_meta : GridMeta
        Metadata returned by `generate_flat_road_surface`.
    pothole_points : (Np,3)
        Pothole points from `generate_bell_pothole_points`.
    pad_cells : int
        Morphological dilation radius (in grid cells) to delete a slightly larger area.

    Returns
    -------
    road_kept : (Nk,3)
        Road points with the region above the pothole removed.
    keep_mask : (Nr,) bool
        Boolean mask such that `road_kept == road_points[keep_mask]`.

    Notes
    -----
    This is O(N) and robust as long as the road is a roughly axis-aligned grid.
    """
    # Map XY -> integer grid indices (i for x, j for y)
    def to_idx(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xi = np.rint((xy[:, 0] - road_meta.x0) /
                     road_meta.spacing).astype(np.int64)
        yi = np.rint((xy[:, 1] - road_meta.y0) /
                     road_meta.spacing).astype(np.int64)
        return xi, yi

    # Indices for pothole points (clamped to grid bounds)
    pi, pj = to_idx(pothole_points[:, :2])
    pi = np.clip(pi, 0, road_meta.nx - 1)
    pj = np.clip(pj, 0, road_meta.ny - 1)

    # Build a 2D deletion mask for the grid
    del_mask = np.zeros((road_meta.ny, road_meta.nx), dtype=bool)
    del_mask[pj, pi] = True

    # Optional dilation in a (2*pad_cells+1)^2 neighborhood
    if pad_cells > 0:
        k = pad_cells
        ny, nx = del_mask.shape
        src = del_mask.copy()
        for dj in range(-k, k + 1):
            y0 = max(0, dj)
            y1 = ny + min(0, dj)
            for di in range(-k, k + 1):
                x0 = max(0, di)
                x1 = nx + min(0, di)
                del_mask[y0:y1, x0:x1] |= src[y0 - dj:y1 - dj, x0 - di:x1 - di]

    # Indices for road points
    ri, rj = to_idx(road_points[:, :2])
    # Outside-of-grid points are kept
    inside = (
        (ri >= 0) & (ri < road_meta.nx) &
        (rj >= 0) & (rj < road_meta.ny)
    )
    keep_mask = np.ones(road_points.shape[0], dtype=bool)
    # Drop points whose (i,j) cell is marked for deletion
    hit = np.zeros_like(keep_mask)
    hit[inside] = del_mask[rj[inside], ri[inside]]
    keep_mask[hit] = False

    return road_points[keep_mask], keep_mask


# ---------------------------------------------------------------------
# 5) Combine road + pothole clouds
# ---------------------------------------------------------------------

def combine_road_and_pothole(
    road_points_kept: np.ndarray,
    pothole_points: np.ndarray,
    return_labels: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Concatenate road & pothole points. Optionally return labels:
    0 = road, 1 = pothole
    """
    if road_points_kept.size == 0 and pothole_points.size == 0:
        pts = np.empty((0, 3), dtype=np.float64)
        return (pts, None) if not return_labels else (pts, np.empty((0,), bool))
    pts = np.vstack([road_points_kept, pothole_points]
                    ).astype(np.float64, copy=False)
    if not return_labels:
        return pts, None
    labels = np.zeros((pts.shape[0],), dtype=np.int8)
    labels[road_points_kept.shape[0]:] = 1
    return pts, labels


# ---------------------------------------------------------------------
# 6) Downward outlier injection (compatible with your test)
# ---------------------------------------------------------------------

def inject_outliers(
    base_points: np.ndarray,
    num_outliers: int = 10,
    z_delta: float = 0.10,
    xy_jitter: float = 0.01,
    z_jitter: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Append `num_outliers` points that are below the base cluster by ~`z_delta`.

    Returns
    -------
    points_with_outliers : (N + num_outliers, 3)
    outlier_mask : (N + num_outliers,) bool
        True for injected outliers, False for original points.

    Behavior
    --------
    - Picks random exemplars from `base_points`, jitters XY by `xy_jitter`,
      and decreases Z by `abs(z_delta)` (plus optional `z_jitter`).
    - Compatible with your existing test that expects a `(points, mask)` pair.
    """
    assert base_points.ndim == 2 and base_points.shape[
        1] == 3, "base_points must be (N,3)"
    g = _rng(seed)

    N = base_points.shape[0]
    if num_outliers <= 0:
        return base_points.astype(np.float64, copy=False), np.zeros((N,), dtype=bool)

    idx = g.integers(0, N, size=num_outliers, endpoint=False)
    samples = base_points[idx].copy()
    if xy_jitter > 0:
        samples[:, :2] += g.normal(0.0, xy_jitter, size=(num_outliers, 2))
    # shift downward
    samples[:, 2] += z_delta
    if z_jitter > 0:
        samples[:, 2] += g.normal(0.0, z_jitter, size=num_outliers)

    pts2 = np.vstack([base_points, samples]).astype(np.float64, copy=False)
    out_mask = np.zeros((pts2.shape[0],), dtype=bool)
    out_mask[N:] = True
    return pts2, out_mask


# ---------------------------------------------------------------------
# 7) One-shot helper to synthesize a full scene (road + carved pothole)
# ---------------------------------------------------------------------

def synthesize_pothole_scene(
    road_nx: int = 300,
    road_ny: int = 200,
    spacing: float = 0.02,
    road_z: float = 0.0,
    pothole_center: Optional[Tuple[float, float]] = None,
    pothole_radius: float = 0.25,
    pothole_depth: float = 0.05,
    pothole_points: int = 4000,
    pad_cells: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, GridMeta]:
    """
    Convenience pipeline that produces a full point cloud with a carved pothole.

    Returns
    -------
    full_cloud : (M,3)
        Combined road + pothole points.
    labels : (M,) int8
        0 = road, 1 = pothole
    meta : GridMeta
        Metadata for the road grid.
    """
    g = _rng(seed)

    # 1) Road
    road, meta = generate_flat_road_surface(
        nx=road_nx, ny=road_ny, spacing=spacing,
        z=road_z, jitter_xy=0.0, jitter_z=0.0, seed=g.integers(1e9)
    )

    # Choose pothole center if not provided (keep away from borders)
    if pothole_center is None:
        margin = 3 * spacing + pothole_radius
        cx = g.uniform(meta.x0 + margin, meta.x1 - margin)
        cy = g.uniform(meta.y0 + margin, meta.y1 - margin)
        pothole_center = (cx, cy)

    # 2–3) Pothole points
    hole, cols = generate_bell_pothole_points(
        center=pothole_center,
        radius=pothole_radius,
        depth=pothole_depth,
        n_points=pothole_points,
        sigma=pothole_radius / 1.75,  # a bit wider than default
        xy_jitter=0.003,
        z_noise=0.002,
        elliptical=0.2,
        tilt=g.normal(0.0, 0.01),
        seed=g.integers(1e9),
        return_colors=True
    )

    # 4) Carve road region above pothole
    road_kept, _ = delete_road_points_above_pothole(
        road_points=road, road_meta=meta, pothole_points=hole, pad_cells=pad_cells
    )

    # 5) Combine
    combined, labels = combine_road_and_pothole(
        road_kept, hole, return_labels=True)
    
    return combined, labels, meta


# ---------------------------------------------------------------------
# Optional: Open3D helpers for convenience (safe to ignore if not using O3D)
# ---------------------------------------------------------------------

def to_open3d(points: np.ndarray, colors: Optional[np.ndarray] = None):
    """Return an Open3D PointCloud from numpy points."""
    return _to_o3d(points, colors)


def write_pcd(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    """Write a PCD/PLY via Open3D if available."""
    if o3d is None:
        raise ImportError(
            "open3d is not installed; cannot write point clouds.")
    pc = _to_o3d(points, colors)
    o3d.io.write_point_cloud(path, pc)

def test_sample(tmp_path):
    # 1) Synthesize a scene (road + pothole points)
    points, labels, meta = synthesize_pothole_scene(seed=42)

    # 2) Inject clear downward outliers appended to the end
    points2, outliers_idx = inject_outliers(points, num_outliers=20, z_delta=0.15, seed=42)

    colors = np.zeros((points2.shape[0], 3), dtype=float)
    existing_pothole_mask = np.zeros((points2.shape[0],), dtype=bool)
    existing_pothole_mask[:labels.shape[0]] = (labels == 1)
    red_mask = existing_pothole_mask | outliers_idx
    colors[red_mask] = [1.0, 0.0, 0.0]

    # 3) Build ground from original road points (labels == 0) and convert to Open3D
    ground_pts = points[labels == 0]
    ground_o3d = to_open3d(ground_pts)

    # 4) Pothole candidates = original pothole points + injected outliers
    pothole_pts = np.vstack([points[labels == 1], points2[outliers_idx]])

    # 5) Run robust filters
    filtered_pts, _, keep_mask = filter_pothole_candidates(pothole_pts, None, ground_o3d, tmp_path=tmp_path, k=10, radius = 2.0, max_below=2.0)

    write_pcd(str(tmp_path / "before_filtering.pcd"), pothole_pts)
    write_pcd(str(tmp_path / "filtered_pts.pcd"), filtered_pts)

    # 6) Assert that all injected outliers (tail segment) are rejected
    num_outliers = int(outliers_idx.sum())
    assert num_outliers > 0
    assert keep_mask.shape[0] == pothole_pts.shape[0]
    assert keep_mask[-num_outliers:].sum() == 0