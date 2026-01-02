"""Ports (Protocol interfaces) for paired-assets analysis.

These define the contracts that infrastructure adapters must implement.
The domain layer depends on these abstractions, not concrete implementations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from paired_assets_analysis.domain.model import RoadScene, RoadSceneRef, SceneInspection


# ---------------------------------------------------------------------------
# Scene Source Port
# ---------------------------------------------------------------------------


class RoadSceneSource(Protocol):
    """Port: enumerate and load road scenes (paired asset folders)."""

    def list(self, *, folder: Path) -> List[RoadSceneRef]:
        """List all valid scenes in the given folder."""
        ...

    def load(self, *, ref: RoadSceneRef) -> RoadScene:
        """Load a scene from its reference."""
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
# Point Cloud Backend Port
# ---------------------------------------------------------------------------


class PointCloudBackend(Protocol):
    """Port: point-cloud processing backend.

    Isolates the domain from concrete libraries (Open3D/NumPy/SciPy/sklearn)
    and IO concerns (reading `.pcd` from disk).
    """

    # IO / Loading
    def read_point_cloud(self, *, pcd_path: Path) -> Any:
        """Read a point cloud file from disk."""
        ...

    def point_count(self, *, pcd: Any) -> int:
        """Return the number of points in a point cloud."""
        ...

    # Preprocessing / Classification
    def classify_points_by_color(
        self,
        *,
        pcd: Any,
        red_threshold: float,
    ) -> Tuple[Any, Any]:
        """Split points into (pothole_points, road_points) by color."""
        ...

    # Surface Estimation
    def segment_plane_road(
        self,
        *,
        pcd: Any,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
    ) -> Tuple[List[float], Any]:
        """Fit a plane to the road surface via RANSAC.

        Returns (plane_model, inlier_mask).
        """
        ...

    # Detection Primitives
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

    def cluster_points(
        self,
        *,
        points: Any,
        depths: Any,
        labels: Any,
        cluster_id: int,
    ) -> Tuple[Any, Any]:
        """Extract points and depths for a specific cluster."""
        ...

    # Metrics Primitives
    def per_pothole_summary(
        self,
        *,
        points: Any,
        depths: Any,
        plane_model: List[float],
    ) -> Dict[str, Any]:
        """Compute summary metrics for a single pothole cluster."""
        ...

    def overall_depth_stats(self, *, depths: Any) -> Optional[Dict[str, float]]:
        """Compute overall depth statistics (max, mean, median)."""
        ...
