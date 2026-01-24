"""Ports (Protocol interfaces) for paired-assets analysis.

These define the contracts that infrastructure adapters must implement.
The domain layer depends on these abstractions, not concrete implementations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from paired_assets_analysis.domain.model import (
    MetricsReport,
    PotholeDetection,
    PreprocessedPointCloud,
    RoadScene,
    SceneInspection,
    PointCloud,
    SurfaceModel,
    SurfaceEstimationConfig,
)


# ---------------------------------------------------------------------------
# Scene Source Port
# ---------------------------------------------------------------------------


class RoadSceneSource(Protocol):
    """Port: enumerate road scenes (paired asset folders)."""

    def list(self, *, folder: Path) -> List[RoadScene]:
        """List all valid scenes in the given folder."""
        ...


# ---------------------------------------------------------------------------
# Repository Port
# ---------------------------------------------------------------------------


class SceneInspectionRepository(Protocol):
    """Port: persist analysis results."""

    def save(self, *, result: SceneInspection) -> None:
        """Save an inspection result."""
        ...


# ---------------------------------------------------------------------------
# Artifacts Repository Port
# ---------------------------------------------------------------------------


class SceneArtifactsRepository(Protocol):
    """Port: persist intermediate artifacts (e.g. ground inlier point clouds).

    This is intentionally separate from SceneInspectionRepository because artifacts
    are binary/heavy and optional.
    """

    def save_ground_inliers(
        self,
        *,
        scene: RoadScene,
        source_pcd: PointCloud,
        inlier_mask: Any,
    ) -> None:
        """Save the surface/ground inlier points derived from `source_pcd`.

        `inlier_mask` is aligned with `source_pcd.points` (same length).
        """
        ...


class PotholeInstancesExporter(Protocol):
    """Port: export one folder per detected pothole instance (cluster).

    This is an optional “dataset shaping” step. It is intentionally separate from
    SceneInspectionRepository because it:
    - copies assets (images)
    - writes cropped point clouds
    - writes one inspection JSON per cluster
    """

    def export(
        self,
        *,
        scene: RoadScene,
        preprocessed: PreprocessedPointCloud,
        detection: PotholeDetection,
        report: MetricsReport,
        surface: SurfaceModel,
    ) -> None:
        """Export per-cluster pothole folders for this scene."""
        ...


# ---------------------------------------------------------------------------
# Point Cloud Backend Port
# ---------------------------------------------------------------------------

class IOPointCloudBackend(Protocol):
    """Port: IO primitives for reading point clouds from disk."""

    def read_point_cloud(self, *, pcd_path: Path) -> PointCloud:
        """Read a point cloud file from disk.

        Returns a domain `PointCloud` object (converted from any concrete format).
        """
        ...


class PointCloudFormatHandler(Protocol):
    """Port: operations that depend on a concrete point-cloud *format/library*.

    Example: Open3D provides a native `segment_plane` implementation. This port
    allows using that capability without leaking Open3D types into the domain.
    """

    def segment_plane_road(
        self,
        *,
        pcd: PointCloud,
        config: SurfaceEstimationConfig,
    ) -> Tuple[List[float], Any]:
        """Estimate the road surface model as a plane.

        Returns (plane_model, inlier_mask).
        """
        ...


class PotholeDetectionHandler(Protocol):
    """Port: numeric primitives used during pothole detection (NumPy/sklearn).
    They're needed to go from candidate pothole points and a surface model to a set of pothole clusters.
    """

    def compute_depths_from_plane(
        self,
        *,
        points: Any,
        plane_model: List[float],
    ) -> Any:
        """Compute signed distances from points to the plane."""
        ...

    def filter_pothole_depths(
        self,
        *,
        points: Any,
        signed_depths: Any,
        threshold: float = 0.0,
    ) -> Tuple[Any, Any]:
        """Filter to points below the plane; return (filtered_points, depths)."""
        ...

    def dbscan_labels(
        self,
        *,
        points: Any,
        eps: float,
        min_samples: int = 10,
    ) -> Tuple[Any, int]:
        """Cluster points with DBSCAN; return (labels, n_clusters)."""
        ...


class PotholeMetricsHandler(Protocol):
    """Port: numeric primitives used during pothole metrics estimation.

    They're needed to go from a set of pothole clusters and a surface model to a set of pothole metrics (area/volume/etc). 
    """

    def per_pothole_summary(
        self,
        *,
        points: Any,
        depths: Any,
        plane_model: List[float],
    ) -> Dict[str, Any]:
        """Compute summary metrics for a single pothole cluster."""
        ...
