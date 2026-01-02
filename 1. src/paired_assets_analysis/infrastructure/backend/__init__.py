"""Low-level point cloud processing utilities (migrated from legacy pcdtools)."""
from paired_assets_analysis.infrastructure.backend._io import (
    read_point_cloud,
    classify_points_by_color,
)
from paired_assets_analysis.infrastructure.backend._plane import (
    segment_plane_road,
    compute_depths_from_plane,
    filter_pothole_depths,
)
from paired_assets_analysis.infrastructure.backend._cluster import dbscan_labels
from paired_assets_analysis.infrastructure.backend._analysis import per_pothole_summary

__all__ = [
    "read_point_cloud",
    "classify_points_by_color",
    "segment_plane_road",
    "compute_depths_from_plane",
    "filter_pothole_depths",
    "dbscan_labels",
    "per_pothole_summary",
]

