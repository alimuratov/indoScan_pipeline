"""Infrastructure adapter implementing the PointCloudBackend port."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paired_assets_analysis.ports import PointCloudBackend


@dataclass(frozen=True)
class PcdToolsBackend(PointCloudBackend):
    """Concrete PointCloudBackend using the backend utilities."""

    # -------------------------------------------------------------------------
    # IO / Loading
    # -------------------------------------------------------------------------

    def read_point_cloud(self, *, pcd_path: Path) -> Any:
        from paired_assets_analysis.infrastructure.backend import read_point_cloud

        return read_point_cloud(str(pcd_path))

    def point_count(self, *, pcd: Any) -> int:
        pts = getattr(pcd, "points", None)
        if pts is None:
            raise TypeError("Expected a point cloud with a .points attribute")
        return len(pts)

    # -------------------------------------------------------------------------
    # Preprocessing / Classification
    # -------------------------------------------------------------------------

    def classify_points_by_color(
        self,
        *,
        pcd: Any,
        red_threshold: float,
    ) -> Tuple[Any, Any]:
        from paired_assets_analysis.infrastructure.backend import classify_points_by_color

        return classify_points_by_color(pcd, red_threshold=red_threshold)

    # -------------------------------------------------------------------------
    # Surface Estimation
    # -------------------------------------------------------------------------

    def segment_plane_road(
        self,
        *,
        pcd: Any,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
    ) -> Tuple[List[float], Any]:
        from paired_assets_analysis.infrastructure.backend import segment_plane_road

        plane_model, inlier_mask = segment_plane_road(
            pcd,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        return [float(x) for x in plane_model], inlier_mask

    # -------------------------------------------------------------------------
    # Detection Primitives
    # -------------------------------------------------------------------------

    def compute_depths_from_plane(
        self,
        *,
        points: Any,
        plane_model: List[float],
    ) -> Any:
        import numpy as np

        from paired_assets_analysis.infrastructure.backend import compute_depths_from_plane

        return compute_depths_from_plane(points, np.asarray(plane_model, dtype=float))

    def filter_pothole_depths(
        self,
        *,
        points: Any,
        signed_depths: Any,
        threshold: float = 0.0,
    ) -> Tuple[Any, Any]:
        from paired_assets_analysis.infrastructure.backend import filter_pothole_depths

        return filter_pothole_depths(points, signed_depths, threshold=threshold)

    def dbscan_labels(
        self,
        *,
        points: Any,
        eps: float,
        min_samples: int = 10,
    ) -> Tuple[Any, int]:
        from paired_assets_analysis.infrastructure.backend import dbscan_labels

        labels, n_clusters = dbscan_labels(
            points, eps=eps, min_samples=min_samples)
        return labels, int(n_clusters)

    def cluster_points(
        self,
        *,
        points: Any,
        depths: Any,
        labels: Any,
        cluster_id: int,
    ) -> Tuple[Any, Any]:
        """Extract points and depths for a specific cluster."""
        mask = labels == cluster_id
        return points[mask], depths[mask]

    # -------------------------------------------------------------------------
    # Metrics Primitives
    # -------------------------------------------------------------------------

    def per_pothole_summary(
        self,
        *,
        points: Any,
        depths: Any,
        plane_model: List[float],
    ) -> Dict[str, Any]:
        import numpy as np

        from paired_assets_analysis.infrastructure.backend import per_pothole_summary

        return per_pothole_summary(
            points,
            depths,
            np.asarray(plane_model, dtype=float),
            compute_surface=True,
        )

    def overall_depth_stats(self, *, depths: Any) -> Optional[Dict[str, float]]:
        """Compute overall depth statistics (max, mean, median)."""
        if depths is None:
            return None
        try:
            if len(depths) == 0:
                return None
        except TypeError:
            return None

        import numpy as np

        return {
            "max_depth": float(np.max(depths)),
            "mean_depth": float(np.mean(depths)),
            "median_depth": float(np.median(depths)),
        }
