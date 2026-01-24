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
class RoadScene:
    """A single scene on disk represented by its `.pcd` path.

    Notes:
    - The scene folder is derived from the `.pcd` parent directory.
    - We keep this lightweight; point-cloud loading happens in preprocessing.
    - It's supposed to be a reference to an input scene, not the loaded data, 
    which allows us to list lots of scenes without loading them into memory.
    """

    id: SceneId
    pcd_path: PcdPath

    @property
    def folder(self) -> Path:
        return self.pcd_path.value.parent


@dataclass(frozen=True)
class PointCloud:
    """Domain representation of a point cloud.

    We keep it library-agnostic: infrastructure adapters are responsible for
    converting Open3D/PCL/etc into this type.

    Notes:
    - `points` is expected to behave like an (N, 3) array (supports `len()` and
      indexing).
    - `colors` is optional and, when present, should align with `points`.
    """

    # if you call PointCloud() with no args, it will create a new empty list for points, so that PointCloud().is_empty() works
    points: Any = field(default_factory=list)
    colors: Optional[Any] = None

    def size(self) -> int:
        try:
            return len(self.points)
        except TypeError:
            return 0

    def get_points(self) -> Any:
        return self.points

    def get_colors(self) -> Optional[Any]:
        return self.colors

    def is_empty(self) -> bool:
        return self.size() == 0


@dataclass(frozen=True)
class PreprocessedPointCloud:
    """Result of preprocessing / loading a scene point cloud.

    Notes:
    - We keep the *loaded* point clouds as domain `PointCloud` objects.
    - We do not store the original combined cloud; if needed, it can be derived
      from `non_pothole_points` + `pothole_points` (ordering may differ from the original).
    """

    pothole_points: PointCloud
    non_pothole_points: PointCloud
    raw_point_count: int

    @property
    def pcd(self) -> PointCloud:
        """Derived combined point cloud (non-pothole + pothole).

        Notes:
        - This does *not* guarantee the original point ordering.
        - Colors are included only if both clouds have colors.
        """

        class _ConcatView:
            def __init__(self, a: Any, b: Any):
                self._a = a
                self._b = b
                try:
                    self._len_a = len(a)
                except TypeError:
                    self._len_a = 0
                try:
                    self._len_b = len(b)
                except TypeError:
                    self._len_b = 0

            def __len__(self) -> int:
                return int(self._len_a + self._len_b)

            def __getitem__(self, idx: Any) -> Any:
                if isinstance(idx, slice):
                    start, stop, step = idx.indices(len(self))
                    return [self[i] for i in range(start, stop, step)]
                if idx < 0:
                    idx = len(self) + idx
                if idx < self._len_a:
                    return self._a[idx]
                return self._b[idx - self._len_a]

        pts_a = self.non_pothole_points.get_points()
        pts_b = self.pothole_points.get_points()

        cols_a = self.non_pothole_points.get_colors()
        cols_b = self.pothole_points.get_colors()
        colors = _ConcatView(cols_a, cols_b) if (
            cols_a is not None and cols_b is not None) else None

        return PointCloud(points=_ConcatView(pts_a, pts_b), colors=colors)


@dataclass(frozen=True)
class PreprocessOutcome:
    status: InspectionStatus
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
    status: InspectionStatus
    data: Optional[PotholeDetection] = None


@dataclass(frozen=True)
class MetricsReport:
    clusters: List[Dict[str, Any]]
    overall: Optional[Dict[str, Any]] = None
    aggregate: Optional[Dict[str, Any]] = None


class Config:
    """Base class for config objects (Value Objects).

    We keep config objects explicit per use-case (e.g. SurfaceEstimationConfig),
    but share a tiny common API:
    - `validate()`: raise if config is structurally invalid
    - `get_params()`: return a shallow copy of params as a plain dict

    Why this shape?
    - Mirrors the repo's general approach: simple dataclasses for configs
      (see legacy `pcdtools.pipeline.*Config` and `scripts/common/config.py`).
    - Avoids `**kwargs` ports while keeping configs extensible via `params`.
    """

    params: Dict[str, Any]

    "Structural validation of the config object. Checks if the params exist, is a dict and keys are strings."

    def validate(self) -> None:
        params = getattr(self, "params", None)
        if params is None:
            raise TypeError(f"{type(self).__name__} is missing `params`")
        if not isinstance(params, dict):
            raise TypeError(f"{type(self).__name__}.params must be a dict")
        for k in params.keys():
            if not isinstance(k, str):
                raise TypeError(
                    f"{type(self).__name__}.params keys must be str (got {type(k).__name__})"
                )

    def get_params(self) -> Dict[str, Any]:
        # Note: even frozen dataclasses can still hold a mutable dict; always copy.
        self.validate()
        return dict(self.params)


@dataclass(frozen=True)
class SurfaceEstimationConfig(Config):
    """Configuration for surface estimation / extraction.

    Why a config object?
    - The port method can stay stable while different implementations
      (RANSAC / Patchwork++ / etc.) use different parameters.
    - Keeps door open without forcing `**kwargs` in the port.
    """

    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SurfaceModel:
    """Road surface model.

    For current integration, this is typically a plane model (a,b,c,d).
    """

    model: List[float]
    method: str
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
