"""Domain layer for segment_processing context."""

from segment_processing.domain.model import (
    SegmentId,
    Segment,
    VideoArtifact,
    VerticalDisplacementEntry,
    VerticalDisplacementSeries,
    RouteLengthResult,
    DepthTimelineEntry,
    DepthTimeline,
    GpsEntry,
    GpsSeries,
    SegmentMeta,
    SegmentProcessingStatus,
    SegmentProcessingResult,
)

__all__ = [
    "SegmentId",
    "Segment",
    "VideoArtifact",
    "VerticalDisplacementEntry",
    "VerticalDisplacementSeries",
    "RouteLengthResult",
    "DepthTimelineEntry",
    "DepthTimeline",
    "GpsEntry",
    "GpsSeries",
    "SegmentMeta",
    "SegmentProcessingStatus",
    "SegmentProcessingResult",
]
