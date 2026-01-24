"""Port interfaces for segment_processing context.

Ports are Protocol interfaces that define what the domain/application layers
need from infrastructure. Adapters implement these protocols.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol

from segment_processing.domain.model import (
    DepthTimeline,
    GpsSeries,
    RouteLengthResult,
    Segment,
    SegmentProcessingResult,
    VerticalDisplacementSeries,
    VideoArtifact,
)


class VideoCreator(Protocol):
    """Port: create survey video from timestamped images.

    Side effect: writes video file to output_path.
    """

    def create_video(
        self,
        *,
        images_dir: Path,
        output_path: Path,
        fps: int,
    ) -> VideoArtifact:
        """Create video from images and return artifact reference."""
        ...


class ImuProcessor(Protocol):
    """Port: convert IMU text log to vertical displacement series.

    Side effect: writes JSON file to output_path.
    """

    def create_series(
        self,
        *,
        imu_file_path: Path,
        output_path: Path,
        interval_seconds: float,
    ) -> VerticalDisplacementSeries:
        """Process IMU file and return series with path."""
        ...


class RouteLengthCalculator(Protocol):
    """Port: calculate route length from odometry log.

    Side effect: writes JSON file to output_path.
    """

    def calculate(
        self,
        *,
        odometry_file_path: Path,
        output_path: Path,
    ) -> RouteLengthResult:
        """Calculate route length and return result with path."""
        ...


class DepthTimelineExtractor(Protocol):
    """Port: extract pothole depth timeline aligned to video.

    Reads pothole inspection_result.json files and aligns depths to video timestamps
    using the earliest timestamp in raw_images as t0.

    Side effect: writes JSON file to output_path.
    """

    def extract(
        self,
        *,
        segment_dir: Path,
        raw_images_dir: Path,
        output_path: Path,
    ) -> DepthTimeline:
        """Extract depth timeline and return result with path."""
        ...


class GpsConverter(Protocol):
    """Port: convert GPS text log to structured JSON.

    Side effect: writes JSON file to output_path.
    """

    def convert(
        self,
        *,
        gps_file_path: Path,
        output_path: Path,
    ) -> GpsSeries:
        """Convert GPS file and return series with path."""
        ...


class SegmentWorkspace(Protocol):
    """Port: provides paths for segment processing.

    Encapsulates the knowledge of where inputs live and where outputs go.
    """

    # Source paths (read from)
    @property
    def segment_dir(self) -> Path:
        """Source segment directory."""
        ...

    @property
    def raw_images_dir(self) -> Path:
        """Directory containing timestamped images for video."""
        ...

    @property
    def imu_txt(self) -> Path:
        """Path to imu.txt file."""
        ...

    @property
    def gps_txt(self) -> Path:
        """Path to gps.txt file."""
        ...

    @property
    def pothole_dirs(self) -> List[Path]:
        """List of pothole_* directories under segment_dir."""
        ...

    # Output paths (write to)
    @property
    def output_video_path(self) -> Path:
        """Where to write output_video.mp4."""
        ...

    @property
    def imu_json_path(self) -> Path:
        """Where to write imu.json."""
        ...

    @property
    def route_length_json_path(self) -> Path:
        """Where to write route_length.json."""
        ...

    @property
    def depth_timeline_json_path(self) -> Path:
        """Where to write segment_depth_timestamps.json."""
        ...

    @property
    def gps_json_path(self) -> Path:
        """Where to write segment_gps.json."""
        ...

    @property
    def segment_meta_json_path(self) -> Path:
        """Where to write segment_meta.json."""
        ...


class SegmentWorkspaceFactory(Protocol):
    """Port: builds SegmentWorkspace from a source reference."""

    def build(self, *, segment_dir: Path) -> SegmentWorkspace:
        """Build workspace for the given segment directory."""
        ...


class SegmentArtifactsRepository(Protocol):
    """Port: persist segment processing results.

    This may write additional derived JSONs (segment_meta.json) or
    copy artifacts to a target tree structure.
    """

    def save(self, *, result: SegmentProcessingResult) -> None:
        """Persist the processing result."""
        ...


class ConsolidatedJsonBuilder(Protocol):
    """Port: build a consolidated roads JSON from a Data2 target tree.

    Side effect: writes JSON file to output_path.
    """

    def build(self, *, target_roads_dir: Path, output_path: Path) -> Path:
        """Build consolidated JSON and return path to the output file."""
        ...
