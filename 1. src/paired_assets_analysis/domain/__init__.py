"""Domain layer: entities, value objects and domain services for paired-assets analysis."""

from .model import (
    SceneId,
    PcdPath,
    RoadSceneRef,
    RoadScene,
    PreprocessedPointCloud,
    PreprocessOutcome,
    SurfaceEstimation,
    PotholeDetection,
    DetectionOutcome,
    MetricsReport,
    SurfaceMethod,
    SurfaceModel,
    PotholeMetrics,
    Pothole,
    InspectionStatus,
    SceneInspection,
)
from .services import (
    PointCloudPreprocessor,
    SurfaceEstimator,
    PotholeDetector,
    PotholeMetricsEstimator,
    SceneAnalysisService,
)
from .strategies import (
    BackendPointCloudPreprocessor,
    BackendRansacSurfaceEstimator,
    BackendPotholeDetector,
    BackendPotholeMetricsEstimator,
)

__all__ = [
    # Value Objects
    "SceneId",
    "PcdPath",
    "RoadSceneRef",
    "RoadScene",
    "PreprocessedPointCloud",
    "PreprocessOutcome",
    "SurfaceEstimation",
    "PotholeDetection",
    "DetectionOutcome",
    "MetricsReport",
    "SurfaceMethod",
    "SurfaceModel",
    "PotholeMetrics",
    "Pothole",
    "InspectionStatus",
    "SceneInspection",
    # Domain Services
    "PointCloudPreprocessor",
    "SurfaceEstimator",
    "PotholeDetector",
    "PotholeMetricsEstimator",
    "SceneAnalysisService",
    # Default strategy implementations (domain, backed by ports)
    "BackendPointCloudPreprocessor",
    "BackendRansacSurfaceEstimator",
    "BackendPotholeDetector",
    "BackendPotholeMetricsEstimator",
]
