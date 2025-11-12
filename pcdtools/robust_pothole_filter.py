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
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(colors, dtype=np.float64))
    return pcd


def _median_abs_dev(x: np.ndarray) -> float:
    """Return the median absolute deviation (MAD).

    Steps:
        1) Compute the median m of `x`.
        2) Compute absolute residuals |x - m| and take their median.

    Args:
        x: 1D array of residuals or measurements.

    Returns:
        The MAD (robust scale estimator).
    """
    m = np.median(x) if x.size else 0.0
    return np.median(np.abs(x - m)) if x.size else 0.0


def _fit_plane_irls(X: np.ndarray, max_iter: int = 3) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return a robustly fitted plane using IRLS.

    Steps:
        1) Initialize equal weights and iterate up to `max_iter` times.
        2) Compute the weighted centroid `mu` and weighted covariance.
        3) Take the eigenvector of the smallest eigenvalue as the plane normal `n`.
        4) Compute signed residuals r = (X - mu)·n and robust scale `sigma` via
           MAD (scaled by 1.4826).
        5) Update Tukey biweight weights from residuals; stop on convergence.

    Args:
        X: (N,3) ground neighborhood used to fit the local plane.
        max_iter: Maximum IRLS iterations.

    Returns:
        n: (3,) unit normal of the plane.
        mu: (3,) a point on the plane (weighted centroid).
        sigma: Robust residual scale (meters) derived from MAD.

    Raises:
        No exceptions are raised; when `N < 3`, a default horizontal plane with
        large `sigma` is returned to signal degeneracy.
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
        """Return neighbor indices by radius or fall back to KNN.

        Args:
            q: Query point (3,) or (1,3).
            k: Number of neighbors for KNN fallback.
            radius: Search radius in meters; if None or empty, use KNN.

        Returns:
            1D array of integer indices into `ref_pts` identifying neighbors.
        """
        q = np.asarray(q, dtype=np.float64).reshape(1, -1)
        if self._backend == "sk":
            if radius is not None and radius > 0:
                idx = self._tree.query_radius(
                    q, r=radius, return_distance=False)[0]
                if idx.size > 0:
                    return idx
            # Fall back to knn
            k_eff = min(k, max(self.ref_pts.shape[0], 1))
            _, idx = self._tree.query(q, k=k_eff)
            return idx[0]
        else:
            # Open3D
            if radius is not None and radius > 0:
                cnt, idx, _ = self._tree.search_radius_vector_3d(
                    q.flatten(), radius)
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
    """Coarse-filter points by distance to a locally fitted ground plane.

    Steps:
        1) For each candidate point `p`, gather nearby ground neighbors by
           `radius` (fallback to KNN with `k`).
        2) If neighbors < `min_neighbors`, reject `p` (cannot define a plane).
        3) Fit a local plane via IRLS to neighbor set; compute signed distance
           `d = (p - mu)·n`.
        4) Form an adaptive robust band = `mad_factor * sigma` (>= 1 mm), cap
           with absolute thresholds `max_above` and `max_below` (more lenient
           below to allow real pothole depth), and keep if `-thr_below <= d <= thr_above`.

    Args:
        pothole_pts: (N,3) candidate points.
        ground_pts: (M,3) road/ground inliers used for local fitting.
        k: KNN fallback size when radius search returns too few neighbors.
        radius: Neighbor search radius in meters; None means KNN only.
        min_neighbors: Minimum ground neighbors required to fit a plane.
        max_above: Absolute cap above plane (meters) for acceptance.
        max_below: Absolute cap below plane (meters) for acceptance.
        mad_factor: Multiplier on robust sigma to form the adaptive band.
        irls_iters: Iterations of IRLS for the local plane fit.
        tmp_path: Optional directory to emit debug artifacts.

    Returns:
        keep_mask: Boolean array over `pothole_pts` marking points to keep.
        signed_distances: Signed distances to the fitted local planes (meters).

    Raises:
        None.
    """
    N = pothole_pts.shape[0]
    keep = np.zeros((N,), dtype=bool)
    dists = np.full((N,), np.nan, dtype=np.float64)

    nn = _NNHelper(ground_pts)

    flag = True

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
            flag = False

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
    """Remove micro-scale outliers using clustering and density tests.

    Steps:
        1) Run DBSCAN to remove isolated speckles.
        2) For each cluster, run Statistical Outlier Removal (SOR).
        3) Run a final radius outlier removal on the kept set.

    Args:
        pothole_pts: (N,3) points after coarse filtering.
        eps: DBSCAN epsilon (meters).
        min_points: DBSCAN minimum samples per cluster.
        sor_neighbors: SOR `nb_neighbors`.
        sor_std: SOR `std_ratio`.
        rad_radius: Radius (meters) for the final radius outlier removal.
        rad_min: Minimum neighbor count for the radius filter.

    Returns:
        Boolean mask over `pothole_pts` for points to keep.

    Raises:
        None.
    """
    if pothole_pts.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    pcd = _as_o3d(pothole_pts)
    labels, n_clusters = dbscan_labels(
        pothole_pts, eps=eps, min_samples=min_points)
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

    pcd3, ind3 = pcd2.remove_radius_outlier(
        nb_points=rad_min, radius=rad_radius)
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
    # (N,3) colors for candidates, or None
    pothole_cols: Optional[np.ndarray],
    # road/ground cloud used to fit local planes
    ground_o3d: o3d.geometry.PointCloud,
    *,
    # coarse
    # KNN fallback size when radius is insufficient
    k: int = 50,
    # neighbor search radius in meters (None => KNN only)
    radius: Optional[float] = 1.0,
    # minimum ground neighbors required to fit a plane
    min_neighbors: int = 15,
    # max meters above plane allowed (reject higher)
    max_above: float = 0.15,
    # max meters below plane allowed (reject deeper)
    max_below: float = 0.35,
    # IRLS robust band multiplier (Tukey biweight)
    mad_factor: float = 3.5,
    irls_iters: int = 3,                     # IRLS iterations for local plane fitting
    # fine
    # DBSCAN epsilon (meters) for cluster cleanup
    micro_eps: float = 0.08,
    micro_min_points: int = 8,               # DBSCAN minimum samples per cluster
    sor_neighbors: int = 20,                 # Statistical Outlier Removal neighbors
    sor_std: float = 2.0,                    # SOR standard deviation ratio threshold
    # radius outlier removal radius (meters)
    rad_radius: float = 0.05,
    rad_min: int = 6,                        # min neighbors within radius to keep a point
    # optional debug output directory for intermediate PCDs
    tmp_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Filter pothole candidates using coarse plane gating and micro cleanup.

    Steps:
        1) Coarse: For each candidate, fit a local plane to nearby ground
           neighbors and accept if the signed distance lies within an adaptive
           robust band capped by `max_above/max_below`.
        2) Fine: Run DBSCAN → SOR → radius outlier removal on the survivors.
        3) Compose the final keep mask back to the original ordering.

    Args:
        pothole_pts: (N,3) candidate pothole points.
        pothole_cols: Optional (N,3) matching RGB colors in [0,1].
        ground_o3d: Ground point cloud providing local neighbors for plane fits.
        k: KNN fallback size for neighbor search.
        radius: Neighbor search radius in meters; None means KNN only.
        min_neighbors: Minimum ground neighbors to fit a plane.
        max_above: Absolute cap above the plane (meters) to keep a point.
        max_below: Absolute cap below the plane (meters) to keep a point.
        mad_factor: IRLS robust band multiplier.
        irls_iters: Number of IRLS iterations.
        micro_eps: DBSCAN epsilon (meters).
        micro_min_points: DBSCAN min samples per cluster.
        sor_neighbors: SOR neighbors.
        sor_std: SOR std ratio.
        rad_radius: Radius outlier removal radius (meters).
        rad_min: Radius outlier minimum neighbor count.
        tmp_path: Optional directory for debug .pcd dumps.

    Returns:
        filtered_pts: (K,3) final points after both stages.
        filtered_cols_or_None: (K,3) colors if provided; else None.
        keep_mask_from_input: (N,) bool mapping to the original input order.

    Raises:
        ValueError: If `ground_o3d` has no points.
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
    cols1 = pothole_cols[keep_far] if (pothole_cols is not None and len(
        pothole_cols) == len(pothole_pts)) else None

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
            _save_pcd(str(tmp_path / "after_micro_filtering2.pcd"),
                      pts1[keep_micro], None)
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
    """Return point and color arrays from a point cloud file.

    Steps:
        1) Read the file with Open3D.
        2) Extract XYZ as float64 and RGB if lengths match points.

    Args:
        path: Input file path (.pcd, .ply, etc.).

    Returns:
        points: (N,3) XYZ array.
        colors: (N,3) RGB array in [0,1] or None when absent/mismatched.
    """
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64) if len(
        pcd.colors) == len(pcd.points) else None
    return pts, cols


def _save_pcd(path: str, pts: np.ndarray, cols: Optional[np.ndarray] = None) -> None:
    """Write a point cloud file from numpy arrays using Open3D.

    Args:
        path: Output file path; extension selects format.
        pts: (N,3) XYZ coordinates.
        cols: Optional (N,3) RGB colors in [0,1].

    Returns:
        None.

    Raises:
        RuntimeError: If Open3D reports a write failure.
    """
    pcd = _as_o3d(pts, cols)
    ok = o3d.io.write_point_cloud(
        path, pcd, write_ascii=False, compressed=False, print_progress=False)
    if not ok:
        raise RuntimeError(f"Failed to save point cloud at {path}")


def main():
    """CLI entrypoint. See module docstring for usage examples."""
    ap = argparse.ArgumentParser(
        description="Robust pothole outlier filtering against a PatchWork++ road surface.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_filter = sub.add_parser(
        "filter", help="Coarse (surface-distance) + fine (micro) filtering.")
    ap_filter.add_argument("--ground_pcd", required=True,
                           help="PCD of ground surface (from PatchWork++).")
    ap_filter.add_argument("--pothole_pcd", required=True,
                           help="PCD of pothole candidates to be filtered.")
    ap_filter.add_argument("--out", required=True,
                           help="Output PCD of filtered pothole points.")
    ap_filter.add_argument("--k", type=int, default=50)
    ap_filter.add_argument("--radius", type=float, default=1.0)
    ap_filter.add_argument("--min_neighbors", type=int, default=15)
    ap_filter.add_argument("--max_above", type=float,
                           default=0.15, help="Meters above plane allowed.")
    ap_filter.add_argument("--max_below", type=float,
                           default=0.35, help="Meters below plane allowed.")
    ap_filter.add_argument("--mad_factor", type=float, default=3.5)
    ap_filter.add_argument("--irls_iters", type=int, default=3)
    ap_filter.add_argument("--micro_eps", type=float, default=0.08)
    ap_filter.add_argument("--micro_min_points", type=int, default=8)
    ap_filter.add_argument("--sor_neighbors", type=int, default=20)
    ap_filter.add_argument("--sor_std", type=float, default=2.0)
    ap_filter.add_argument("--rad_radius", type=float, default=0.05)
    ap_filter.add_argument("--rad_min", type=int, default=6)

    ap_micro = sub.add_parser(
        "micro", help="Only micro cleanup (DBSCAN+SOR+Radius) on a pothole PCD.")
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
        print(
            f"[filter] input={len(pothole_pts)} kept={kept} ({100.0*kept/max(len(pothole_pts),1):.1f}%) saved={args.out}")

    elif args.cmd == "micro":
        pothole_pts, pothole_cols = _load_pcd_xyzrgb(args.pothole_pcd)
        keep_mask = micro_outlier_cleanup(
            pothole_pts,
            eps=args.micro_eps, min_points=args.micro_min_points,
            sor_neighbors=args.sor_neighbors, sor_std=args.sor_std,
            rad_radius=args.rad_radius, rad_min=args.rad_min,
        )
        _save_pcd(args.out, pothole_pts[keep_mask],
                  pothole_cols[keep_mask] if pothole_cols is not None else None)
        print(
            f"[micro] input={len(pothole_pts)} kept={int(keep_mask.sum())} saved={args.out}")

    else:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
