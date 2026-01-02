"""Domain-level strategy implementations that depend only on ports.

These implementations contain orchestration/math policy, but delegate all
library-specific details (Open3D/NumPy/SciPy/sklearn and filesystem IO) to the
`PointCloudBackend` port.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paired_assets_analysis.domain.model import (
    DetectionOutcome,
    InspectionStatus,
    MetricsReport,
    PotholeDetection,
    PreprocessOutcome,
    PreprocessedPointCloud,
    RoadScene,
    SurfaceEstimation,
    SurfaceMethod,
    SurfaceModel,
)
from paired_assets_analysis.domain.services import (
    PointCloudPreprocessor,
    PotholeDetector,
    PotholeMetricsEstimator,
    SurfaceEstimator,
)
from paired_assets_analysis.ports import PointCloudBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_empty(obj: Any) -> bool:
    """Check if an array-like object is empty (has length 0)."""
    try:
        return len(obj) == 0
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# Strategy Implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendPointCloudPreprocessor(PointCloudPreprocessor):
    """Load the PCD and split pothole vs road points by color."""

    backend: PointCloudBackend
    red_threshold: float = 0.7

    def preprocess(self, scene: RoadScene) -> PreprocessOutcome:
        pcd = self.backend.read_point_cloud(pcd_path=scene.pcd_path.value)
        n_points = self.backend.point_count(pcd=pcd)

        if n_points < 1:
            return PreprocessOutcome(status=InspectionStatus.EMPTY_CLOUD)

        pothole_points, road_points = self.backend.classify_points_by_color(
            pcd=pcd,
            red_threshold=self.red_threshold,
        )

        if _is_empty(pothole_points):
            return PreprocessOutcome(status=InspectionStatus.NO_POTHOLE_POINTS)

        return PreprocessOutcome(
            status=InspectionStatus.OK,
            data=PreprocessedPointCloud(
                pcd=pcd,
                pothole_points=pothole_points,
                road_points=road_points,
                raw_point_count=n_points,
            ),
        )


@dataclass(frozen=True)
class BackendRansacSurfaceEstimator(SurfaceEstimator):
    """Estimate road surface as the dominant plane via Open3D RANSAC."""

    backend: PointCloudBackend
    distance_threshold: float = 0.02
    ransac_n: int = 3
    num_iterations: int = 1000

    def estimate(self, preprocessed: PreprocessedPointCloud) -> SurfaceEstimation:
        plane_model, inlier_mask = self.backend.segment_plane_road(
            pcd=preprocessed.pcd,
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
        )
        surface = SurfaceModel(
            model=[float(x) for x in plane_model],
            method=SurfaceMethod.RANSAC,
            metadata={},
        )
        return SurfaceEstimation(surface=surface, inlier_mask=inlier_mask)


@dataclass(frozen=True)
class BackendPotholeDetector(PotholeDetector):
    """Filter pothole points below the plane and cluster with DBSCAN."""

    backend: PointCloudBackend
    eps: float = 0.1
    summary_only: bool = False
    eps_max: float = 1e6
    dbscan_min_samples: int = 10

    def detect(
        self,
        *,
        preprocessed: PreprocessedPointCloud,
        surface: SurfaceModel,
    ) -> DetectionOutcome:
        # Compute signed depths relative to the road plane
        signed_depths = self.backend.compute_depths_from_plane(
            points=preprocessed.pothole_points,
            plane_model=surface.model,
        )

        # Keep only points below the plane
        filtered_points, depths = self.backend.filter_pothole_depths(
            points=preprocessed.pothole_points,
            signed_depths=signed_depths,
            threshold=0.0,
        )

        if _is_empty(filtered_points):
            return DetectionOutcome(status=InspectionStatus.NO_POINTS_BELOW_PLANE)

        # Single-cluster mode: skip DBSCAN
        if self.summary_only:
            return DetectionOutcome(
                status=InspectionStatus.OK,
                data=PotholeDetection(
                    filtered_points=filtered_points,
                    depths=depths,
                    labels=None,
                    n_clusters=1,
                ),
            )

        # Cluster with DBSCAN
        labels, n_clusters = self.backend.dbscan_labels(
            points=filtered_points,
            eps=self.eps,
            min_samples=self.dbscan_min_samples,
        )

        # Fallback: if no clusters found, use a very large epsilon
        if n_clusters == 0:
            labels, n_clusters = self.backend.dbscan_labels(
                points=filtered_points,
                eps=self.eps_max,
                min_samples=self.dbscan_min_samples,
            )

        return DetectionOutcome(
            status=InspectionStatus.OK,
            data=PotholeDetection(
                filtered_points=filtered_points,
                depths=depths,
                labels=labels,
                n_clusters=n_clusters,
            ),
        )


@dataclass(frozen=True)
class BackendPotholeMetricsEstimator(PotholeMetricsEstimator):
    """Compute per-cluster and overall metrics (convex hull + TIN volumes)."""

    backend: PointCloudBackend
    aggregate_all: bool = False  # kept for parity; not used in current MetricsReport

    def estimate(
        self,
        *,
        detection: PotholeDetection,
        surface: SurfaceModel,
    ) -> MetricsReport:
        clusters = self._compute_cluster_summaries(detection, surface)
        overall = self.backend.overall_depth_stats(depths=detection.depths)
        return MetricsReport(clusters=clusters, overall=overall, aggregate=None)

    def _compute_cluster_summaries(
        self,
        detection: PotholeDetection,
        surface: SurfaceModel,
    ) -> list:
        """Compute per-pothole summary for each cluster."""
        # Single-cluster mode (no labels)
        if detection.n_clusters == 1 and detection.labels is None:
            return [
                self.backend.per_pothole_summary(
                    points=detection.filtered_points,
                    depths=detection.depths,
                    plane_model=surface.model,
                )
            ]

        # Multi-cluster mode
        summaries = []
        for cluster_id in range(detection.n_clusters):
            pts, dps = self.backend.cluster_points(
                points=detection.filtered_points,
                depths=detection.depths,
                labels=detection.labels,
                cluster_id=cluster_id,
            )
            summaries.append(
                self.backend.per_pothole_summary(
                    points=pts,
                    depths=dps,
                    plane_model=surface.model,
                )
            )
        return summaries
