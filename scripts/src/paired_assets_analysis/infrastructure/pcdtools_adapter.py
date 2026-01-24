"""Infrastructure adapters implementing paired-assets-analysis ports.

This module intentionally keeps adapters small and focused:
- IO: read a file and convert into domain `PointCloud`
- Format handler: operations that require a concrete point-cloud library (Open3D)
- Numeric handlers: NumPy/sklearn/scipy based primitives for detection/metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from paired_assets_analysis.ports import (
    IOPointCloudBackend,
    PointCloudFormatHandler,
    PotholeDetectionHandler,
    PotholeMetricsHandler,
)
from paired_assets_analysis.domain.model import PointCloud
from paired_assets_analysis.domain.model import SurfaceEstimationConfig


class Open3DIOBackend(IOPointCloudBackend):
    def read_point_cloud(self, *, pcd_path: Path) -> PointCloud:
        import open3d as o3d

        open3d_pcd = o3d.io.read_point_cloud(str(pcd_path))

        points = np.asarray(open3d_pcd.points)
        colors = (
            np.asarray(open3d_pcd.colors)
            if len(open3d_pcd.colors) == len(open3d_pcd.points) and len(open3d_pcd.colors) > 0
            else None
        )

        return PointCloud(
            points=points,
            colors=colors,
        )

    def convert_PointCloud_to_Open3D(self, *, pcd: PointCloud) -> Any:
        import open3d as o3d

        o3d_pcd = o3d.geometry.PointCloud()
        pts = np.asarray(pcd.get_points(), dtype=np.float64)
        o3d_pcd.points = o3d.utility.Vector3dVector(pts)

        cols = pcd.get_colors()
        if cols is not None:
            cols_arr = np.asarray(cols, dtype=np.float64)
            if len(cols_arr) == len(pts) and cols_arr.size > 0:
                o3d_pcd.colors = o3d.utility.Vector3dVector(cols_arr)

        return o3d_pcd

    def convert_Open3D_to_PointCloud(self, *, pcd: Any) -> PointCloud:
        return PointCloud(
            points=np.asarray(pcd.points),
            colors=np.asarray(pcd.colors) if len(pcd.colors) == len(
                pcd.points) and len(pcd.colors) > 0 else None,
        )


@dataclass(frozen=True)
class Open3DPointCloudFormatHandler(PointCloudFormatHandler):
    """Point-cloud format handler (Patchwork++ ground extraction + Open3D RANSAC plane fit)."""

    io_backend: Open3DIOBackend

    def segment_plane_road(
        self,
        *,
        pcd: PointCloud,
        config: SurfaceEstimationConfig,
    ) -> Tuple[List[float], Any]:
        if pcd.is_empty():
            raise ValueError("Empty point cloud")

        config.validate()

        params = config.get_params()

        import pypatchworkpp
        import open3d as o3d

        pts = np.asarray(pcd.get_points(), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                "PointCloud.points must be an array-like of shape (N, 3)")
        pts = np.ascontiguousarray(pts, dtype=np.float32)

        # 1) Patchwork++ ground extraction
        pw_params = pypatchworkpp.Parameters()
        for k, v in params.items():
            if hasattr(pw_params, k):
                try:
                    setattr(pw_params, k, v)
                except Exception:
                    # pybind attributes can be picky about types; ignore if it can't be set
                    pass

        pp = pypatchworkpp.patchworkpp(pw_params)
        pp.estimateGround(pts)

        ground_idx = np.asarray(pp.getGroundIndices(), dtype=int)
        if ground_idx.size == 0:
            raise ValueError("Patchwork++ returned no ground indices")

        ground_pts = pts[ground_idx]
        if ground_pts.shape[0] < 3:
            raise ValueError(
                "Patchwork++ returned too few ground points to fit a plane")

        # 2) Open3D RANSAC plane fit on extracted ground points
        distance_threshold = float(params.get("distance_threshold", 0.02))
        ransac_n = int(params.get("ransac_n", 3))
        num_iterations = int(params.get("num_iterations", 1000))

        ground_o3d = o3d.geometry.PointCloud()
        ground_o3d.points = o3d.utility.Vector3dVector(
            np.asarray(ground_pts, dtype=np.float64)
        )
        plane_model, ransac_inliers = ground_o3d.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        plane_model = np.asarray(plane_model, dtype=float)
        if plane_model.shape[0] >= 3 and plane_model[2] < 0:
            plane_model = -plane_model

        # RANSAC inliers are indices into `ground_pts`; map back to original point indices.
        ransac_inliers = np.asarray(ransac_inliers, dtype=int)
        inlier_mask = np.zeros((len(pts),), dtype=bool)
        if ransac_inliers.size > 0:
            inlier_mask[ground_idx[ransac_inliers]] = True

        return [float(x) for x in plane_model], inlier_mask


@dataclass(frozen=True)
class NumpyPotholeDetectionHandler(PotholeDetectionHandler):
    """Detection primitives backed by NumPy + sklearn (via infrastructure backend utils)."""

    @staticmethod
    def _compute_depths_from_plane(points: Any, plane_model: Any) -> np.ndarray:
        """Compute signed distance of points to plane.

        Positive above plane, negative below plane.
        Plane is ax + by + cz + d = 0.
        """
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            return np.zeros((0,), dtype=float)
        plane = np.asarray(plane_model, dtype=float)
        a, b, c, d = plane
        n = np.array([a, b, c], dtype=float)
        n_norm = np.linalg.norm(n) + 1e-12
        return (pts @ n + d) / n_norm

    @staticmethod
    def _filter_pothole_depths(
        points: Any,
        signed_depths: Any,
        *,
        threshold: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Keep only points below the plane and return positive depths."""
        pts = np.asarray(points, dtype=float)
        depths = np.asarray(signed_depths, dtype=float)
        if depths.size == 0 or pts.size == 0:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
        mask = depths < float(threshold)
        return pts[mask], np.abs(depths[mask])

    @staticmethod
    def _dbscan_labels(
        points: Any,
        *,
        eps: float,
        min_samples: int = 10,
    ) -> Tuple[np.ndarray, int]:
        """Cluster 3D points with DBSCAN. Returns (labels, n_clusters)."""
        from sklearn.cluster import DBSCAN

        pts = np.asarray(points, dtype=float)
        if len(pts) == 0:
            return np.array([]), 0
        clustering = DBSCAN(
            eps=float(eps), min_samples=int(min_samples)).fit(pts)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return labels, int(n_clusters)

    def compute_depths_from_plane(
        self,
        *,
        points: Any,
        plane_model: List[float],
    ) -> Any:
        return self._compute_depths_from_plane(points, np.asarray(plane_model, dtype=float))

    def filter_pothole_depths(
        self,
        *,
        points: Any,
        signed_depths: Any,
        threshold: float = 0.0,
    ) -> Tuple[Any, Any]:
        return self._filter_pothole_depths(points, signed_depths, threshold=threshold)

    def dbscan_labels(
        self,
        *,
        points: Any,
        eps: float,
        min_samples: int = 10,
    ) -> Tuple[Any, int]:
        labels, n_clusters = self._dbscan_labels(
            points, eps=eps, min_samples=min_samples)
        return labels, n_clusters


@dataclass(frozen=True)
class NumpyPotholeMetricsHandler(PotholeMetricsHandler):
    """Metrics primitives backed by NumPy (+ scipy inside infrastructure backend utils)."""

    @staticmethod
    def _convex_hull_area(points_xy: np.ndarray) -> Tuple[float, Optional[object]]:
        """Return (area, hull_object) of the convex hull in 2D.

        For 2D, scipy ConvexHull.volume equals polygon area.
        """
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(points_xy)
            return float(hull.volume), hull
        except Exception:
            return 0.0, None

    @staticmethod
    def _z_on_plane(plane: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Evaluate plane ax + by + cz + d = 0 as z(x, y) on XY points."""
        a, b, c, d = plane
        if abs(float(c)) < 1e-12:
            raise ValueError("Plane nearly vertical; cannot express z(x, y)")
        return -(a * xy[:, 0] + b * xy[:, 1] + d) / c

    @classmethod
    def _tin_volume_over_points(
        cls,
        points: np.ndarray,
        road_plane: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute volume under the road plane using a TIN over the pothole points.

        Returns (volume, max_depth).
        """
        if points.shape[0] < 3:
            return 0.0, 0.0

        xy = points[:, :2]
        z = points[:, 2]

        # Triangulate
        try:
            from scipy.spatial import Delaunay

            tri = Delaunay(xy)
            triangles = np.asarray(tri.simplices, dtype=np.int32)
        except Exception:
            try:
                import matplotlib.tri as mtri

                triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
                triangles = np.asarray(triang.triangles, dtype=np.int32)
            except Exception:
                return 0.0, 0.0

        if triangles.size == 0:
            return 0.0, 0.0

        # Depths at vertices relative to road plane
        try:
            z_road = cls._z_on_plane(road_plane, xy)
        except Exception:
            return 0.0, 0.0
        depth = z_road - z

        vol = 0.0
        max_depth = float(np.max(np.maximum(depth, 0.0))
                          ) if depth.size else 0.0
        for i, j, k in triangles:
            v0, v1, v2 = xy[i], xy[j], xy[k]
            # XY area of the triangle
            area = 0.5 * \
                abs(np.linalg.det(np.stack([v1 - v0, v2 - v0], axis=0)))
            d0, d1, d2 = depth[i], depth[j], depth[k]
            # Clamp to only integrate below-plane regions
            d0 = max(d0, 0.0)
            d1 = max(d1, 0.0)
            d2 = max(d2, 0.0)
            vol += area * (d0 + d1 + d2) / 3.0

        return float(vol), float(max_depth)

    @classmethod
    def _per_pothole_summary(
        cls,
        points: Any,
        depths: Any,
        road_plane: Optional[np.ndarray] = None,
        *,
        compute_surface: bool = False,
    ) -> Dict[str, Any]:
        """Produce a compact set of metrics for a pothole cluster."""
        pts = np.asarray(points, dtype=float)
        d = np.asarray(depths, dtype=float)

        summary: Dict[str, Any] = {
            "points": int(len(pts)),
            "max_depth": float(d.max()) if len(d) else 0.0,
            "mean_depth": float(d.mean()) if len(d) else 0.0,
            "median_depth": float(np.median(d)) if len(d) else 0.0,
        }
        hull_area, _ = cls._convex_hull_area(
            pts[:, :2]) if pts.shape[0] else (0.0, None)
        summary["hull_area"] = float(hull_area)
        summary["simple_volume"] = float(
            hull_area * (d.mean() if len(d) else 0.0))

        if compute_surface and road_plane is not None:
            vol_tin, _max_depth_tin = cls._tin_volume_over_points(
                pts, road_plane)
            summary["delaunay_volume"] = float(vol_tin)
            if summary["simple_volume"] > 0:
                summary["volume_ratio_delaunay_over_convex"] = float(
                    vol_tin / summary["simple_volume"]
                )

        return summary

    def per_pothole_summary(
        self,
        *,
        points: Any,
        depths: Any,
        plane_model: List[float],
    ) -> Dict[str, Any]:
        return self._per_pothole_summary(
            points,
            depths,
            np.asarray(plane_model, dtype=float),
            compute_surface=True,
        )
