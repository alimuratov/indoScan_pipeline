from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SceneId:
    value: str


@dataclass(frozen=True)
class PcdPath:
    value: Path


@dataclass(frozen=True)
class RoadSceneRef:
    """Reference to a scene on disk (paired asset folder)."""

    id: SceneId
    pcd_path: PcdPath
    folder: Path


@dataclass(frozen=True)
class RoadScene:
    """Loaded scene. For now we keep it lightweight: just identity + pcd path."""

    id: SceneId
    pcd_path: PcdPath
    folder: Path


@dataclass(frozen=True)
class PreprocessedPointCloud:
    """Result of preprocessing / loading a scene point cloud.

    Notes:
    - We intentionally keep these fields typed as `Any` to avoid coupling the
      domain layer to NumPy/Open3D types.
    """

    pcd: Any
    pothole_points: Any
    road_points: Any
    raw_point_count: int


@dataclass(frozen=True)
class PreprocessOutcome:
    status: "InspectionStatus"
    data: Optional[PreprocessedPointCloud] = None


@dataclass(frozen=True)
class SurfaceEstimation:
    surface: "SurfaceModel"
    inlier_mask: Any = None


@dataclass(frozen=True)
class PotholeDetection:
    """Result of pothole detection (filtered points + clustering).

    Notes:
    - `labels` is None in summary_only mode (single cluster).
    - `n_clusters` is always >= 1 when status is OK.
    """

    filtered_points: Any
    depths: Any
    labels: Optional[Any]  # None in summary_only mode
    n_clusters: int


@dataclass(frozen=True)
class DetectionOutcome:
    status: "InspectionStatus"
    data: Optional[PotholeDetection] = None


@dataclass(frozen=True)
class MetricsReport:
    clusters: List[Dict[str, Any]]
    overall: Optional[Dict[str, Any]] = None
    aggregate: Optional[Dict[str, Any]] = None


class SurfaceMethod(str, Enum):
    RANSAC = "ransac"
    PATCHWORKPP = "patchworkpp"


@dataclass(frozen=True)
class SurfaceModel:
    """Road surface model.

    For current integration, this is typically a plane model (a,b,c,d).
    """

    model: List[float]
    method: SurfaceMethod
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PotholeMetrics:
    n_points: int
    max_depth: float
    mean_depth: float
    median_depth: float
    hull_area: float
    volume_convex: float
    volume_delaunay: Optional[float] = None


@dataclass(frozen=True)
class Pothole:
    """A single detected pothole cluster."""

    cluster_id: int
    metrics: PotholeMetrics


class InspectionStatus(str, Enum):
    OK = "ok"
    EMPTY_CLOUD = "empty_cloud"
    NO_POTHOLE_POINTS = "no_pothole_points"
    NO_POINTS_BELOW_PLANE = "no_points_below_plane"
    FAILED = "failed"


@dataclass(frozen=True)
class SceneInspection:
    scene: RoadScene
    status: InspectionStatus
    surface: Optional[SurfaceModel] = None
    potholes: List[Pothole] = field(default_factory=list)
    overall: Optional[Dict[str, Any]] = None  # keep flexible for now
