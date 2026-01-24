"""Domain model for segment_processing context.

Contains entities, value objects, and the aggregate result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Value Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentId:
    """Unique identifier for a segment (e.g., folder name or UUID)."""

    value: str


@dataclass(frozen=True)
class VideoArtifact:
    """Reference to a generated video file."""

    path: Path
    fps: int
    frame_count: int = 0
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class VerticalDisplacementEntry:
    """Single vertical displacement sample."""

    video_timestamp: str  # Relative seconds as string (legacy format)
    vertical_displacement: float


@dataclass(frozen=True)
class VerticalDisplacementSeries:
    """Collection of vertical displacement samples."""

    entries: List[VerticalDisplacementEntry]
    path: Optional[Path] = None  # Where it was written (if persisted)


@dataclass(frozen=True)
class RouteLengthResult:
    """Calculated route length."""

    meters: float
    kilometers: float
    path: Optional[Path] = None  # Where it was written (if persisted)


@dataclass(frozen=True)
class DepthTimelineEntry:
    """Single pothole depth entry aligned to video timeline."""

    video_timestamp: str  # Relative seconds as string
    pothole_depth: float


@dataclass(frozen=True)
class DepthTimeline:
    """Collection of pothole depth entries aligned to video timeline."""

    entries: List[DepthTimelineEntry]
    path: Optional[Path] = None  # Where it was written (if persisted)


@dataclass(frozen=True)
class GpsEntry:
    """Single GPS entry."""

    timestamp: float
    lat: float
    lng: float
    alt: Optional[float] = None


@dataclass(frozen=True)
class GpsSeries:
    """Collection of GPS entries for a segment."""

    entries: List[GpsEntry]
    path: Optional[Path] = None  # Where it was written (if persisted)

    @property
    def start_location(self) -> str:
        """Return 'lat, lng' of first entry or empty string."""
        if not self.entries:
            return ""
        e = self.entries[0]
        return f"{e.lat}, {e.lng}"

    @property
    def end_location(self) -> str:
        """Return 'lat, lng' of last entry or empty string."""
        if not self.entries:
            return ""
        e = self.entries[-1]
        return f"{e.lat}, {e.lng}"


@dataclass(frozen=True)
class SegmentMeta:
    """Aggregated segment metadata (route summary)."""

    start_loc: str
    end_loc: str
    length_in_km: float
    path: Optional[Path] = None  # Where it was written (if persisted)


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """Domain entity representing a road segment."""

    id: SegmentId
    source_dir: Path

    @property
    def name(self) -> str:
        return self.source_dir.name


# ---------------------------------------------------------------------------
# Processing Result (Aggregate)
# ---------------------------------------------------------------------------


class SegmentProcessingStatus(Enum):
    """Status of segment processing."""

    OK = "ok"
    PARTIAL = "partial"  # Some artifacts missing but not fatal
    FAILED = "failed"


@dataclass
class SegmentProcessingResult:
    """Aggregate result of processing a segment.

    Contains all produced artifacts and metadata. Artifacts are optional
    because processing may partially succeed.
    """

    segment: Segment
    status: SegmentProcessingStatus

    video: Optional[VideoArtifact] = None
    imu_series: Optional[VerticalDisplacementSeries] = None
    route_length: Optional[RouteLengthResult] = None
    depth_timeline: Optional[DepthTimeline] = None
    gps_series: Optional[GpsSeries] = None
    segment_meta: Optional[SegmentMeta] = None

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
