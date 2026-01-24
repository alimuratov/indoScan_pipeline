"""Domain services (and strategy protocols) for paired-assets analysis."""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

from paired_assets_analysis.ports import PotholeInstancesExporter, SceneArtifactsRepository

from .model import (
    DetectionOutcome,
    InspectionStatus,
    MetricsReport,
    Pothole,
    PotholeMetrics,
    PotholeDetection,
    PreprocessOutcome,
    PreprocessedPointCloud,
    RoadScene,
    SceneInspection,
    SurfaceEstimation,
    SurfaceModel,
    PointCloud,
)


# ---------------------------------------------------------------------------
# Strategy Protocols
# ---------------------------------------------------------------------------


class PointCloudPreprocessor(Protocol):
    """Strategy port: load and preprocess the scene point cloud."""

    def preprocess(self, scene: RoadScene) -> PreprocessOutcome: ...

    def _split_pothole_and_non_pothole_points_by_red_color(
        self, pcd: PointCloud, red_threshold) -> Tuple[Any, Any]: ...


class SurfaceEstimator(Protocol):
    """Strategy port: estimate the road surface model (e.g. plane)."""

    def estimate(
        self, preprocessed: PreprocessedPointCloud) -> SurfaceEstimation: ...

    def compute_surface_model(self, *, pcd: PointCloud,
                              **kwargs) -> SurfaceModel: ...


class PotholeDetector(Protocol):
    """Strategy port: filter pothole candidates + cluster into potholes."""

    def detect(
        self,
        *,
        preprocessed: PreprocessedPointCloud,
        surface: SurfaceModel,
    ) -> DetectionOutcome: ...


class PotholeMetricsEstimator(Protocol):
    """Strategy port: compute pothole metrics from detections."""

    def estimate(
        self,
        *,
        detection: PotholeDetection,
        surface: SurfaceModel,
    ) -> MetricsReport: ...


# ---------------------------------------------------------------------------
# Orchestration Service
# ---------------------------------------------------------------------------


class SceneAnalysisService:
    """Domain service: orchestrates scene analysis.

    Orchestrates four strategy protocols:
    - PointCloudPreprocessor
    - SurfaceEstimator
    - PotholeDetector
    - PotholeMetricsEstimator
    """

    def __init__(
        self,
        *,
        preprocessor: PointCloudPreprocessor,
        surface_estimator: SurfaceEstimator,
        pothole_detector: PotholeDetector,
        metrics_estimator: PotholeMetricsEstimator,
        artifacts_repo: SceneArtifactsRepository | None = None,
        pothole_exporter: PotholeInstancesExporter | None = None,
    ) -> None:
        self._preprocessor = preprocessor
        self._surface_estimator = surface_estimator
        self._pothole_detector = pothole_detector
        self._metrics_estimator = metrics_estimator
        self._artifacts_repo = artifacts_repo
        self._pothole_exporter = pothole_exporter

    def analyze(self, scene: RoadScene) -> SceneInspection:
        """Analyze the given scene and return a SceneInspection."""

        # Step 1: Preprocess
        match self._preprocessor.preprocess(scene):
            case PreprocessOutcome(status=InspectionStatus.OK, data=PreprocessedPointCloud() as pre):
                pass
            case PreprocessOutcome(status=fail_status):
                return SceneInspection(scene=scene, status=fail_status)

        # Step 2: Surface estimation
        try:
            surface_est = self._surface_estimator.estimate(pre)
        except Exception:
            return SceneInspection(scene=scene, status=InspectionStatus.FAILED)

        surface = surface_est.surface

        # Optional: persist intermediate artifacts (e.g. surface/ground inlier cloud)
        if self._artifacts_repo is not None:
            self._artifacts_repo.save_ground_inliers(
                scene=scene,
                source_pcd=pre.non_pothole_points,
                inlier_mask=surface_est.inlier_mask,
            )

        # Step 3: Pothole detection
        match self._pothole_detector.detect(preprocessed=pre, surface=surface):
            case DetectionOutcome(status=InspectionStatus.OK, data=PotholeDetection() as detection):
                pass
            case DetectionOutcome(status=fail_status):
                # Legacy behavior: return surface even if detection fails
                return SceneInspection(scene=scene, status=fail_status, surface=surface)

        # Step 4: Metrics estimation
        try:
            report = self._metrics_estimator.estimate(
                detection=detection, surface=surface)
        except Exception:
            return SceneInspection(scene=scene, status=InspectionStatus.FAILED, surface=surface)

        # Optional: export one folder per pothole cluster (crop + copy assets + per-cluster JSON)
        if self._pothole_exporter is not None:
            self._pothole_exporter.export(
                scene=scene,
                preprocessed=pre,
                detection=detection,
                report=report,
                surface=surface,
            )

        # Map raw cluster dicts to domain Pothole objects
        potholes = [
            Pothole(cluster_id=idx, metrics=_parse_cluster_metrics(cluster_data))
            for idx, cluster_data in enumerate(report.clusters or [])
        ]

        overall = _merge_aggregate(report.overall, report.aggregate)

        return SceneInspection(
            scene=scene,
            status=InspectionStatus.OK,
            surface=surface,
            potholes=potholes,
            overall=overall,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_cluster_metrics(cluster_data: Dict[str, Any]) -> PotholeMetrics:
    """Parse a raw cluster dict into a PotholeMetrics value object."""
    return PotholeMetrics(
        n_points=int(cluster_data.get("points", 0)),
        max_depth=float(cluster_data.get("max_depth", 0.0)),
        mean_depth=float(cluster_data.get("mean_depth", 0.0)),
        median_depth=float(cluster_data.get("median_depth", 0.0)),
        hull_area=float(cluster_data.get("hull_area", 0.0)),
        volume_convex=float(cluster_data.get("simple_volume", 0.0)),
        volume_delaunay=(
            float(cluster_data["delaunay_volume"])
            if "delaunay_volume" in cluster_data
            else None
        ),
    )


def _merge_aggregate(
    overall: Dict[str, Any] | None,
    aggregate: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    """Merge aggregate stats into overall dict if present."""
    if aggregate is None:
        return overall
    merged = dict(overall or {})
    merged["aggregate"] = aggregate
    return merged
