"""Domain-level strategy implementations that depend only on ports.

These implementations contain orchestration/math policy, but delegate all
library-specific details (Open3D/NumPy/SciPy/sklearn and filesystem IO) to
infrastructure adapters via ports.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Tuple, Optional, Dict

from paired_assets_analysis.domain.model import (
    DetectionOutcome,
    InspectionStatus,
    MetricsReport,
    PotholeDetection,
    PreprocessOutcome,
    PreprocessedPointCloud,
    PointCloud,
    RoadScene,    SurfaceEstimation,
    SurfaceEstimationConfig,
    SurfaceModel,
)
from paired_assets_analysis.domain.services import (
    PointCloudPreprocessor,
    PotholeDetector,
    PotholeMetricsEstimator,
    SurfaceEstimator,
)
from paired_assets_analysis.ports import (
    IOPointCloudBackend,
    PointCloudFormatHandler,
    PotholeDetectionHandler,
    PotholeMetricsHandler,
)


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
class DefaultPointCloudPreprocessor(PointCloudPreprocessor):
    """Load the PCD and split pothole vs non-pothole points by color."""

    io_backend: IOPointCloudBackend
    red_threshold: float = 0.7

    def preprocess(self, scene: RoadScene) -> PreprocessOutcome:
        pcd = self.io_backend.read_point_cloud(pcd_path=scene.pcd_path.value)
        n_points = pcd.size()

        if n_points < 1:
            return PreprocessOutcome(status=InspectionStatus.EMPTY_CLOUD)

        pothole_points_pcd, non_pothole_points_pcd = self._split_pothole_and_non_pothole_points_by_red_color(
            pcd=pcd,
            red_threshold=self.red_threshold,
        )

        if pothole_points_pcd.is_empty():
            return PreprocessOutcome(status=InspectionStatus.NO_POTHOLE_POINTS)

        return PreprocessOutcome(
            status=InspectionStatus.OK,
            data=PreprocessedPointCloud(
                pothole_points=pothole_points_pcd,
                non_pothole_points=non_pothole_points_pcd,
                raw_point_count=n_points,
            ),
        )

    def _split_pothole_and_non_pothole_points_by_red_color(
            self, pcd: PointCloud, red_threshold: float = 0.7) -> Tuple[PointCloud, PointCloud]:
        """Split points into pothole vs road using a simple red-channel heuristic.

        - Pothole points: R > red_threshold and G,B < 0.3
        - Non-pothole points: complement

        Returns (pothole_pcd, non_pothole_pcd) as domain PointCloud objects.
        If no color is present, all points are treated as road and potholes empty.
        """
        if pcd.size() == 0:
            return PointCloud(), PointCloud()

        points, colors = pcd.get_points(), pcd.get_colors()
        if colors is None:
            return PointCloud(), PointCloud(points=points, colors=None)

        red_mask = (
            (colors[:, 0] > red_threshold)
            & (colors[:, 1] < 0.3)
            & (colors[:, 2] < 0.3)
        )
        road_mask = ~red_mask
        pothole_points = points[red_mask]
        non_pothole_points = points[road_mask]
        return PointCloud(points=pothole_points, colors=colors[red_mask]), PointCloud(points=non_pothole_points, colors=colors[road_mask])


@dataclass(frozen=True)
class BackendPatchworkPPSurfaceEstimator(SurfaceEstimator):
    """Estimate road surface using Patchwork++ ground extraction + plane fit.

    This strategy runs Patchwork++ on the *non-red* points (preprocessed.non_pothole_points),
    then fits a single global plane to the extracted ground points.
    """

    format_handler: PointCloudFormatHandler
    uprightness_thr: float = 0.95
    max_range: float = 1000.0
    num_iter: int = 20

    def estimate(self, preprocessed: PreprocessedPointCloud) -> SurfaceEstimation:
        cfg = SurfaceEstimationConfig(
            params={
                # Patchwork++ params (names depend on build; unknown params are ignored by the adapter)
                "uprightness_thr": self.uprightness_thr,
                "max_range": self.max_range,
                "num_iter": self.num_iter,
            },
        )
        plane_model, inlier_mask = self.format_handler.segment_plane_road(
            pcd=preprocessed.non_pothole_points,
            config=cfg,
        )
        surface = SurfaceModel(
            model=[float(x) for x in plane_model],
            method="patchworkpp+ransac",
            metadata={"params": cfg.get_params()},
        )
        return SurfaceEstimation(surface=surface, inlier_mask=inlier_mask)


@dataclass(frozen=True)
class DefaultPotholeDetector(PotholeDetector):
    """Filter pothole points below the plane and cluster with DBSCAN."""

    detection: PotholeDetectionHandler
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
        pothole_points = preprocessed.pothole_points.get_points()

        # Compute signed depths relative to the road plane
        signed_depths = self.detection.compute_depths_from_plane(
            points=pothole_points,
            plane_model=surface.model,
        )

        # Keep only points below the plane
        filtered_points, depths = self.detection.filter_pothole_depths(
            points=pothole_points,
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
        labels, n_clusters = self.detection.dbscan_labels(
            points=filtered_points,
            eps=self.eps,
            min_samples=self.dbscan_min_samples,
        )

        # Fallback: if no clusters found, use a very large epsilon
        if n_clusters == 0:
            labels, n_clusters = self.detection.dbscan_labels(
                points=filtered_points,
                eps=self.eps_max,
                min_samples=self.dbscan_min_samples,
            )

        # If DBSCAN still produced zero clusters (e.g. too few points for min_samples),
        # treat the entire set as a single cluster (consistent with summary_only semantics).
        if n_clusters == 0:
            return DetectionOutcome(
                status=InspectionStatus.OK,
                data=PotholeDetection(
                    filtered_points=filtered_points,
                    depths=depths,
                    labels=None,
                    n_clusters=1,
                ),
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

    metrics: PotholeMetricsHandler
    aggregate_all: bool = False  # kept for parity; not used in current MetricsReport

    def estimate(
        self,
        *,
        detection: PotholeDetection,
        surface: SurfaceModel,
    ) -> MetricsReport:
        clusters = self._compute_cluster_summaries(detection, surface)
        overall = _overall_depth_stats(detection.depths)
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
                self.metrics.per_pothole_summary(
                    points=detection.filtered_points,
                    depths=detection.depths,
                    plane_model=surface.model,
                )
            ]

        # Multi-cluster mode
        summaries = []
        for cluster_id in range(detection.n_clusters):
            mask = detection.labels == cluster_id
            pts = detection.filtered_points[mask]
            dps = detection.depths[mask]
            summaries.append(
                self.metrics.per_pothole_summary(
                    points=pts,
                    depths=dps,
                    plane_model=surface.model,
                )
            )
        return summaries


def _overall_depth_stats(depths: Any) -> Optional[Dict[str, float]]:
    """Compute overall depth statistics (max, mean, median) in domain.

    This is pure data reduction over an array-like `depths`, so it doesn't need
    a port / infrastructure adapter.
    """
    if depths is None:
        return None
    try:
        n = len(depths)
    except TypeError:
        return None
    if n == 0:
        return None

    vals = [float(x) for x in depths]
    return {
        "max_depth": float(max(vals)),
        "mean_depth": float(sum(vals) / n),
        "median_depth": float(median(vals)),
    }
