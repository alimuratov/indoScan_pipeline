#!/usr/bin/env python3
"""Robust two-scale outlier removal for pothole points.

This module filters pothole candidate points using a coarse surface-distance
gate against a locally fitted ground plane (IRLS), and then a fine micro
cleanup (DBSCAN + SOR + radius). Includes a CLI with two subcommands:

Usage (CLI examples):
  # Filter pothole candidates against a ground surface and save the filtered set
  python robust_pothole_filter.py filter \
      --ground_pcd ground_surface.pcd \
      --pothole_pcd pothole_candidates.pcd \
      --out filtered_potholes.pcd

  # Run only the fine-scale cleanup on a pothole set (DBSCAN + SOR + radius)
  python robust_pothole_filter.py micro \
      --pothole_pcd pothole_candidates.pcd \
      --out filtered_potholes_micro.pcd

Import use (inside your existing pipeline):
  from robust_pothole_filter import filter_pothole_candidates

  filtered_pts, filtered_cols, keep_mask = filter_pothole_candidates(
      pothole_pts, pothole_cols, ground_o3d,
      k=50, radius=1.0, min_neighbors=15,
      max_above=0.15, max_below=0.35,
      mad_factor=3.5, irls_iters=3,
      micro_eps=0.08, micro_min_points=8,
      sor_neighbors=20, sor_std=2.0,
      rad_radius=0.05, rad_min=6
  )
"""

from __future__ import annotations
import argparse
import sys
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import open3d as o3d

from pcdtools.cluster import dbscan_labels
from pcdtools.io import write_point_cloud_from_arrays

try:
    # Faster neighbor search if available
    from sklearn.neighbors import KDTree as SkKDTree
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# --------------------------- Helpers ---------------------------

def _as_o3d(points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    """Build an Open3D PointCloud from numpy arrays.

    Args:
        points: Array of shape (N, 3) with XYZ coordinates.
        colors: Optional array of shape (N, 3) with RGB in [0, 1].

    Returns:
        o3d.geometry.PointCloud with points (and colors if provided).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def _median_abs_dev(x: np.ndarray) -> float:
    """Robust scale: MAD = median(|x - median(x)|)."""
    m = np.median(x) if x.size else 0.0
    return np.median(np.abs(x - m)) if x.size else 0.0


def _fit_plane_irls(X: np.ndarray, max_iter: int = 3) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Robust plane fit via IRLS with Tukey's biweight.
    Returns (normal n (unit), point_on_plane mu, robust_sigma).
    """
    eps = 1e-12
    if X.shape[0] < 3:
        # Degenerate: return a dummy plane with huge sigma
        return np.array([0.0, 0.0, 1.0]), X.mean(axis=0), 1e6

    w = np.ones((X.shape[0],), dtype=np.float64)

    for _ in range(max_iter):
        W = w[:, None]
        denom = np.sum(W) + eps
        mu = (W * X).sum(axis=0) / denom
        Xc = X - mu
        # Weighted covariance
        C = (Xc * W).T @ Xc / max(denom - 1.0, 1.0)
        # Eigenvector for smallest eigenvalue -> normal
        evals, evecs = np.linalg.eigh(C)
        n = evecs[:, 0]
        n = n / (np.linalg.norm(n) + eps)

        # Residuals (signed distances along n)
        r = Xc @ n
        mad = _median_abs_dev(r)
        sigma = 1.4826 * mad  # Consistent estimator for Gaussian
        if sigma < 1e-9:
            # Perfect plane fit or tiny scale -> stop
            break

        c = 4.685 * sigma  # Tukey constant
        u = r / (c + eps)
        w_new = (1.0 - u**2) ** 2
        w_new[np.abs(u) >= 1.0] = 0.0

        if np.allclose(w, w_new, atol=1e-6, rtol=1e-6):
            w = w_new
            break
        w = w_new

    return n, mu, float(sigma)


class _NNHelper:
    """Neighbor search helper. Uses sklearn KDTree if present, else Open3D KDTreeFlann."""
    def __init__(self, ref_pts: np.ndarray):
        self.ref_pts = np.asarray(ref_pts, dtype=np.float64)
        if _HAS_SKLEARN:
            self._tree = SkKDTree(self.ref_pts, leaf_size=40)
            self._backend = "sk"
            self._o3d = None
        else:
            self._o3d_pcd = _as_o3d(self.ref_pts)
            self._tree = o3d.geometry.KDTreeFlann(self._o3d_pcd)
            self._backend = "o3d"

    def radius_or_knn(self, q: np.ndarray, k: int, radius: Optional[float]) -> np.ndarray:
        """Return neighbor indices within radius if available, otherwise KNN."""
        q = np.asarray(q, dtype=np.float64).reshape(1, -1)
        if self._backend == "sk":
            if radius is not None and radius > 0:
                idx = self._tree.query_radius(q, r=radius, return_distance=False)[0]
                if idx.size > 0:
                    return idx
            # Fall back to knn
            k_eff = min(k, max(self.ref_pts.shape[0], 1))
            _, idx = self._tree.query(q, k=k_eff)
            return idx[0]
        else:
            # Open3D
            if radius is not None and radius > 0:
                cnt, idx, _ = self._tree.search_radius_vector_3d(q.flatten(), radius)
                if cnt > 0:
                    return np.asarray(idx, dtype=int)
            cnt, idx, _ = self._tree.search_knn_vector_3d(q.flatten(), k)
            return np.asarray(idx, dtype=int)


# --------------------------- Core algorithms ---------------------------

def filter_by_ground_surface(
    pothole_pts: np.ndarray,
    ground_pts: np.ndarray,
    *,
    k: int = 50,
    radius: Optional[float] = 1.0,
    min_neighbors: int = 15,
    max_above: float = 0.15,
    max_below: float = 0.35,
    mad_factor: float = 3.5,
    irls_iters: int = 3,
    tmp_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Coarse filter against a locally fitted ground plane.

    For each red candidate point, look at the nearby ground points and fit a local road patch (a small plane). 
    Keep the point only if its distance to that plane is small.
    -> Things like taillights and sign reflections are far above the road -> rejected
    -> Real pothole points lie just below the road -> allowed

    Returns:
        keep_mask: Boolean mask over pothole_pts to keep.
        signed_distances: Signed distances n·(p - mu); positive means above plane.

    Notes:
        - max_above (m): cap above the plane (tail-lights, poles, etc.).
        - max_below (m): allowed depth below the plane (deep cavities).
    """
    N = pothole_pts.shape[0]
    keep = np.zeros((N,), dtype=bool)
    dists = np.full((N,), np.nan, dtype=np.float64)

    nn = _NNHelper(ground_pts)

    flag=True

    for i in range(N):
        p = pothole_pts[i]
        idx = nn.radius_or_knn(p, k=k, radius=radius)
        if idx.size < min_neighbors:
            # Too few ground neighbors to define a plane -> reject
            continue

        X = ground_pts[idx]
        # n is a unit plane normal
        # mu is a point on the plane
        # sigma = 1.4826 × MAD(r) where r represents the distances between the plane and the ground points
        n, mu, sigma = _fit_plane_irls(X, max_iter=irls_iters)

        d = float((p - mu) @ n)  # signed distance
        dists[i] = d

        # Robust, adaptive gate + absolute caps
        # lower (below) side gets a bit more allowance
        robust_band = mad_factor * max(sigma, 1e-3)  # >= 1 mm
        thr_above = min(max_above, robust_band)
        thr_below = min(max_below, robust_band * 20)
        # print(f"thr_above: {thr_above}, thr_below: {thr_below}, robust_band: {robust_band}, d: {d}")

        if -thr_below <= d <= thr_above:
            keep[i] = True
        elif flag:
            # write the plane points based on which the point is rejected
            # write_point_cloud_from_arrays(str(tmp_path / "after_coarse_filtering_but_rejected.pcd"), X, None)
            flag=False

    return keep, dists


def micro_outlier_cleanup(
    pothole_pts: np.ndarray,
    *,
    eps: float = 0.08,
    min_points: int = 8,
    sor_neighbors: int = 20,
    sor_std: float = 2.0,
    rad_radius: float = 0.05,
    rad_min: int = 6,
) -> np.ndarray:
    """
    On the candidates that survive the geometry gate, run simple, 
    local density cleanups to shave off tiny speckles or stray points without erasing the real pothole.

    Steps:
        1) DBSCAN to drop isolated speckles.
        2) Statistical Outlier Removal (SOR).
        3) Radius Outlier Removal (tight radius).

    Returns:
        Boolean mask over pothole_pts for points to keep.
    """
    if pothole_pts.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    pcd = _as_o3d(pothole_pts)
    labels, n_clusters = dbscan_labels(pothole_pts, eps=eps, min_samples=min_points)
    keep0 = labels != -1
    if not np.any(keep0):
        return np.zeros_like(keep0)

    # keep only the points that passed the DBSCAN
    pcd1 = pcd.select_by_index(np.where(keep0)[0])

    labels1 = labels[keep0]
    keep1 = np.zeros((len(pcd1.points),), dtype=bool)

    for c_id in sorted(set(labels1)):
        # labels1 was built from keep0, so -1 shouldn't appear; guard anyway
        if c_id == -1:
            continue

        cluster_idx = np.where(labels1 == c_id)[0]
        if cluster_idx.size == 0:
            continue
        cluster_sub = pcd1.select_by_index(cluster_idx.tolist())

        # If the cluster is too small for SOR, keep it as-is
        if len(cluster_sub.points) < sor_neighbors:
            keep1[cluster_idx] = True
            continue

        # SOR per cluster
        _, cluster_idx_sor = cluster_sub.remove_statistical_outlier(
            nb_neighbors=sor_neighbors, std_ratio=sor_std
        )
        keep1[cluster_idx[cluster_idx_sor]] = True

    # Radius outlier on the kept set (optional but recommended)
    idx1 = np.where(keep1)[0]
    if idx1.size == 0:
        return np.zeros_like(keep0)
    pcd2 = pcd1.select_by_index(idx1.tolist())

    pcd3, ind3 = pcd2.remove_radius_outlier(nb_points=rad_min, radius=rad_radius)
    keep2 = np.zeros_like(keep1)
    if len(ind3) > 0:
        keep2[idx1[np.asarray(ind3, dtype=int)]] = True

    # Map back to original indices
    idx_keep0 = np.where(keep0)[0]
    idx_keep_final = idx_keep0[keep2]

    final_keep = np.zeros((pothole_pts.shape[0],), dtype=bool)
    final_keep[idx_keep_final] = True
    return final_keep


def filter_pothole_candidates(
    pothole_pts: np.ndarray,                 # (N,3) candidate pothole points
    pothole_cols: Optional[np.ndarray],      # (N,3) colors for candidates, or None
    ground_o3d: o3d.geometry.PointCloud,     # road/ground cloud used to fit local planes
    *,
    # coarse
    k: int = 50,                             # KNN fallback size when radius is insufficient
    radius: Optional[float] = 1.0,           # neighbor search radius in meters (None => KNN only)
    min_neighbors: int = 15,                 # minimum ground neighbors required to fit a plane
    max_above: float = 0.15,                 # max meters above plane allowed (reject higher)
    max_below: float = 0.35,                 # max meters below plane allowed (reject deeper)
    mad_factor: float = 3.5,                 # IRLS robust band multiplier (Tukey biweight)
    irls_iters: int = 3,                     # IRLS iterations for local plane fitting
    # fine
    micro_eps: float = 0.08,                 # DBSCAN epsilon (meters) for cluster cleanup
    micro_min_points: int = 8,               # DBSCAN minimum samples per cluster
    sor_neighbors: int = 20,                 # Statistical Outlier Removal neighbors
    sor_std: float = 2.0,                    # SOR standard deviation ratio threshold
    rad_radius: float = 0.05,                # radius outlier removal radius (meters)
    rad_min: int = 6,                        # min neighbors within radius to keep a point
    tmp_path: Optional[Path] = None,         # optional debug output directory for intermediate PCDs
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Two-scale filtering of pothole candidates against PatchWork++ ground.

    Coarse plane distance gating followed by micro outlier cleanup.

    Returns:
        filtered_pts: Final points after both stages.
        filtered_cols_or_None: Matching colors if provided on input.
        keep_mask_from_input: Boolean mask mapping to the original input order.
    """
    ground_pts = np.asarray(ground_o3d.points, dtype=np.float64)
    if ground_pts.size == 0:
        raise ValueError("ground_o3d has no points.")

    # Coarse: reject far-from-surface
    keep_far, dists = filter_by_ground_surface(
        pothole_pts, ground_pts,
        k=k, radius=radius, min_neighbors=min_neighbors,
        max_above=max_above, max_below=max_below,
        mad_factor=mad_factor, irls_iters=irls_iters,
        tmp_path=tmp_path,
    )

    pts1 = pothole_pts[keep_far]
    cols1 = pothole_cols[keep_far] if (pothole_cols is not None and len(pothole_cols) == len(pothole_pts)) else None

    # Optional debug writes when tmp_path provided (avoids test dependency)
    if tmp_path is not None:
        try:
            _save_pcd(str(tmp_path / "after_coarse_filtering2.pcd"), pts1, None)
        except Exception:
            pass
    # Fine: micro cleanup
    keep_micro = micro_outlier_cleanup(
        pts1,
        eps=micro_eps, min_points=micro_min_points,
        sor_neighbors=sor_neighbors, sor_std=sor_std,
        rad_radius=rad_radius, rad_min=rad_min,
    )

    if tmp_path is not None:
        try:
            _save_pcd(str(tmp_path / "after_micro_filtering2.pcd"), pts1[keep_micro], None)
        except Exception:
            pass

    pts2 = pts1[keep_micro]
    cols2 = cols1[keep_micro] if cols1 is not None else None


    # Compose final mask (relative to original pothole_pts)
    keep_final = np.zeros_like(keep_far)
    idx_far = np.where(keep_far)[0]
    keep_final[idx_far[keep_micro]] = True

    return pts2, cols2, keep_final


# --------------------------- CLI ---------------------------

def _load_pcd_xyzrgb(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a PCD/PLY/etc. into numpy arrays.

    Returns:
        points (N,3), colors (N,3) or None if not present/length-mismatch.
    """
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64) if len(pcd.colors) == len(pcd.points) else None
    return pts, cols


def _save_pcd(path: str, pts: np.ndarray, cols: Optional[np.ndarray] = None) -> None:
    """Save numpy arrays as a point cloud using Open3D."""
    pcd = _as_o3d(pts, cols)
    ok = o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=False, print_progress=False)
    if not ok:
        raise RuntimeError(f"Failed to save point cloud at {path}")


def main():
    """CLI entrypoint. See module docstring for usage examples."""
    ap = argparse.ArgumentParser(description="Robust pothole outlier filtering against a PatchWork++ road surface.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_filter = sub.add_parser("filter", help="Coarse (surface-distance) + fine (micro) filtering.")
    ap_filter.add_argument("--ground_pcd", required=True, help="PCD of ground surface (from PatchWork++).")
    ap_filter.add_argument("--pothole_pcd", required=True, help="PCD of pothole candidates to be filtered.")
    ap_filter.add_argument("--out", required=True, help="Output PCD of filtered pothole points.")
    ap_filter.add_argument("--k", type=int, default=50)
    ap_filter.add_argument("--radius", type=float, default=1.0)
    ap_filter.add_argument("--min_neighbors", type=int, default=15)
    ap_filter.add_argument("--max_above", type=float, default=0.15, help="Meters above plane allowed.")
    ap_filter.add_argument("--max_below", type=float, default=0.35, help="Meters below plane allowed.")
    ap_filter.add_argument("--mad_factor", type=float, default=3.5)
    ap_filter.add_argument("--irls_iters", type=int, default=3)
    ap_filter.add_argument("--micro_eps", type=float, default=0.08)
    ap_filter.add_argument("--micro_min_points", type=int, default=8)
    ap_filter.add_argument("--sor_neighbors", type=int, default=20)
    ap_filter.add_argument("--sor_std", type=float, default=2.0)
    ap_filter.add_argument("--rad_radius", type=float, default=0.05)
    ap_filter.add_argument("--rad_min", type=int, default=6)

    ap_micro = sub.add_parser("micro", help="Only micro cleanup (DBSCAN+SOR+Radius) on a pothole PCD.")
    ap_micro.add_argument("--pothole_pcd", required=True)
    ap_micro.add_argument("--out", required=True)
    ap_micro.add_argument("--micro_eps", type=float, default=0.08)
    ap_micro.add_argument("--micro_min_points", type=int, default=8)
    ap_micro.add_argument("--sor_neighbors", type=int, default=20)
    ap_micro.add_argument("--sor_std", type=float, default=2.0)
    ap_micro.add_argument("--rad_radius", type=float, default=0.05)
    ap_micro.add_argument("--rad_min", type=int, default=6)

    args = ap.parse_args()

    if args.cmd == "filter":
        ground_pts, _ = _load_pcd_xyzrgb(args.ground_pcd)
        pothole_pts, pothole_cols = _load_pcd_xyzrgb(args.pothole_pcd)
        ground_o3d = _as_o3d(ground_pts)

        filtered_pts, filtered_cols, keep_mask = filter_pothole_candidates(
            pothole_pts, pothole_cols, ground_o3d,
            k=args.k, radius=args.radius, min_neighbors=args.min_neighbors,
            max_above=args.max_above, max_below=args.max_below, mad_factor=args.mad_factor,
            irls_iters=args.irls_iters,
            micro_eps=args.micro_eps, micro_min_points=args.micro_min_points,
            sor_neighbors=args.sor_neighbors, sor_std=args.sor_std,
            rad_radius=args.rad_radius, rad_min=args.rad_min,
        )
        _save_pcd(args.out, filtered_pts, filtered_cols)
        kept = int(keep_mask.sum())
        print(f"[filter] input={len(pothole_pts)} kept={kept} ({100.0*kept/max(len(pothole_pts),1):.1f}%) saved={args.out}")

    elif args.cmd == "micro":
        pothole_pts, pothole_cols = _load_pcd_xyzrgb(args.pothole_pcd)
        keep_mask = micro_outlier_cleanup(
            pothole_pts,
            eps=args.micro_eps, min_points=args.micro_min_points,
            sor_neighbors=args.sor_neighbors, sor_std=args.sor_std,
            rad_radius=args.rad_radius, rad_min=args.rad_min,
        )
        _save_pcd(args.out, pothole_pts[keep_mask], pothole_cols[keep_mask] if pothole_cols is not None else None)
        print(f"[micro] input={len(pothole_pts)} kept={int(keep_mask.sum())} saved={args.out}")

    else:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
