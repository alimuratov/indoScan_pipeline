"""Workspace implementation for segment_processing context.

Provides path resolution for segment inputs and outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class FilesystemSegmentWorkspace:
    """Concrete workspace that resolves paths for filesystem-based segments.

    By default, outputs are written to the source segment directory (legacy behavior).
    This can be overridden by passing a different output_dir.
    """

    segment_dir: Path
    raw_images_folder_name: str = "raw_images"
    imu_filename: str = "imu.txt"
    gps_filename: str = "gps.txt"
    pothole_prefix: str = "pothole_"

    # Output filenames (written to segment_dir by default)
    output_video_filename: str = "output_video.mp4"
    imu_json_filename: str = "imu.json"
    route_length_json_filename: str = "route_length.json"
    depth_timeline_json_filename: str = "segment_depth_timestamps.json"
    gps_json_filename: str = "segment_gps.json"
    segment_meta_json_filename: str = "segment_meta.json"

    # Optional: override output directory (for target tree export)
    _output_dir: Path | None = None

    @property
    def output_dir(self) -> Path:
        """Directory where outputs are written."""
        return self._output_dir or self.segment_dir

    # --- Source paths ---

    @property
    def raw_images_dir(self) -> Path:
        """Directory containing timestamped images used for video + timelines.

        Convention:
        - Preferred: `<segment_dir>/raw_images/`
        - Common dataset alias: `<segment_dir>/Images/`
        """
        primary = self.segment_dir / self.raw_images_folder_name
        if primary.is_dir():
            return primary

        # Many datasets store the full timestamped frame stream under `Images/`.
        images_alias = self.segment_dir / "Images"
        if images_alias.is_dir():
            return images_alias

        return primary

    @property
    def imu_txt(self) -> Path:
        return self.segment_dir / self.imu_filename

    @property
    def gps_txt(self) -> Path:
        return self.segment_dir / self.gps_filename

    @property
    def pothole_dirs(self) -> List[Path]:
        """Discover pothole_* directories under segment_dir."""
        if not self.segment_dir.is_dir():
            return []
        return sorted(
            p
            for p in self.segment_dir.iterdir()
            if p.is_dir() and p.name.startswith(self.pothole_prefix)
        )

    # --- Output paths ---

    @property
    def output_video_path(self) -> Path:
        return self.output_dir / self.output_video_filename

    @property
    def imu_json_path(self) -> Path:
        return self.output_dir / self.imu_json_filename

    @property
    def route_length_json_path(self) -> Path:
        return self.output_dir / self.route_length_json_filename

    @property
    def depth_timeline_json_path(self) -> Path:
        return self.output_dir / self.depth_timeline_json_filename

    @property
    def gps_json_path(self) -> Path:
        return self.output_dir / self.gps_json_filename

    @property
    def segment_meta_json_path(self) -> Path:
        return self.output_dir / self.segment_meta_json_filename


@dataclass
class FilesystemSegmentWorkspaceFactory:
    """Factory that builds FilesystemSegmentWorkspace instances."""

    raw_images_folder_name: str = "raw_images"
    imu_filename: str = "imu.txt"
    gps_filename: str = "gps.txt"
    pothole_prefix: str = "pothole_"
    output_dir: Path | None = None  # If set, outputs go here instead of segment_dir

    def build(self, *, segment_dir: Path) -> FilesystemSegmentWorkspace:
        """Build workspace for the given segment directory."""
        return FilesystemSegmentWorkspace(
            segment_dir=segment_dir,
            raw_images_folder_name=self.raw_images_folder_name,
            imu_filename=self.imu_filename,
            gps_filename=self.gps_filename,
            pothole_prefix=self.pothole_prefix,
            _output_dir=self.output_dir,
        )
