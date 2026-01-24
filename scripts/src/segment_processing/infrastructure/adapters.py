"""Infrastructure adapters for segment_processing context.

These adapters implement the port protocols by wrapping legacy functions
or providing new implementations.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from segment_processing.domain.model import (
    DepthTimeline,
    DepthTimelineEntry,
    GpsEntry,
    GpsSeries,
    RouteLengthResult,
    SegmentProcessingResult,
    VerticalDisplacementEntry,
    VerticalDisplacementSeries,
    VideoArtifact,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VideoCreator adapter (wraps legacy create_video.py)
# ---------------------------------------------------------------------------


@dataclass
class OpenCVVideoCreator:
    """Creates video from timestamped images using OpenCV.

    Wraps the legacy media.create_video.create_video_from_images function.
    """

    def create_video(
        self,
        *,
        images_dir: Path,
        output_path: Path,
        fps: int,
    ) -> VideoArtifact:
        """Create video from images and return artifact reference."""
        # Import here to avoid heavy dependency at module load
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError(
                "OpenCV (cv2) is required for video creation") from e

        image_files = sorted(
            f for f in images_dir.iterdir() if f.suffix.lower() == ".jpg"
        )

        if not image_files:
            raise ValueError(f"No .jpg files found in {images_dir}")

        # Sort by timestamp (filename is like "1234567890.123456789.jpg")
        def get_timestamp(p: Path) -> float:
            match = re.match(r"(\d+\.\d+)", p.stem)
            return float(match.group(1)) if match else 0.0

        image_files.sort(key=get_timestamp)

        # Read first image to get dimensions
        first_frame = cv2.imread(str(image_files[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first image: {image_files[0]}")

        height, width = first_frame.shape[:2]

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        try:
            for img_path in image_files:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    logger.warning(
                        "Could not read image: %s, skipping", img_path)
                    continue
                # Resize if needed
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
                frame_count += 1
        finally:
            writer.release()

        duration = frame_count / fps if fps > 0 else 0.0

        logger.info(
            "Created video: %s (%d frames, %.2f seconds)",
            output_path,
            frame_count,
            duration,
        )

        return VideoArtifact(
            path=output_path,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
        )


# ---------------------------------------------------------------------------
# ImuProcessor adapter (wraps legacy process_gps.py logic)
# ---------------------------------------------------------------------------


@dataclass
class LegacyImuProcessor:
    """Processes IMU text log to vertical displacement series.

    Wraps the legacy sensors.process_gps.process_imu_file logic.
    """

    def create_series(
        self,
        *,
        imu_file_path: Path,
        output_path: Path,
        interval_seconds: float = 30.0,
    ) -> VerticalDisplacementSeries:
        """Process IMU file and return series with path."""
        entries: List[VerticalDisplacementEntry] = []

        with imu_file_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        next_time = 0.0
        start_timestamp: Optional[float] = None

        for line in lines:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    timestamp = float(parts[0])
                    z_displacement = float(parts[3])

                    if start_timestamp is None:
                        start_timestamp = timestamp

                    relative_time = timestamp - start_timestamp

                    if relative_time >= next_time:
                        entries.append(
                            VerticalDisplacementEntry(
                                video_timestamp=str(int(next_time)),
                                vertical_displacement=round(z_displacement, 6),
                            )
                        )
                        next_time += interval_seconds
                except ValueError:
                    continue

        # Write JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "vertical_displacement": e.vertical_displacement,
                "video_timestamp": e.video_timestamp,
            }
            for e in entries
        ]
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        logger.info("Wrote IMU series: %s (%d entries)",
                    output_path, len(entries))

        return VerticalDisplacementSeries(entries=entries, path=output_path)


# ---------------------------------------------------------------------------
# RouteLengthCalculator adapter (wraps legacy calculate_road_length.py)
# ---------------------------------------------------------------------------


@dataclass
class LegacyRouteLengthCalculator:
    """Calculates route length from odometry file.

    Wraps the legacy sensors.calculate_road_length.calculate_route_length logic.
    """

    def calculate(
        self,
        *,
        odometry_file_path: Path,
        output_path: Path,
    ) -> RouteLengthResult:
        """Calculate route length and return result with path."""
        try:
            import numpy as np
        except ImportError as e:
            raise RuntimeError(
                "NumPy is required for route length calculation") from e

        positions: List[List[float]] = []

        with odometry_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] != "#" and len(parts) >= 4:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        positions.append([x, y, z])
                    except ValueError:
                        continue

        if len(positions) < 2:
            logger.warning("Not enough points for route length calculation")
            meters = 0.0
        else:
            pos_array = np.array(positions)
            segment_distances = np.linalg.norm(
                np.diff(pos_array, axis=0), axis=1)
            meters = float(np.sum(segment_distances))

        kilometers = meters / 1000.0

        # Write JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"meters": round(meters, 6),
                       "kilometers": round(kilometers, 6)}, indent=2),
            encoding="utf-8",
        )

        logger.info("Wrote route length: %s (%.2f m)", output_path, meters)

        return RouteLengthResult(meters=meters, kilometers=kilometers, path=output_path)


# ---------------------------------------------------------------------------
# DepthTimelineExtractor adapter (wraps legacy segment_depth_timestamps.py)
# ---------------------------------------------------------------------------


@dataclass
class LegacyDepthTimelineExtractor:
    """Extracts pothole depth timeline aligned to video.

    Reads inspection_result.json from pothole folders (paired_assets_analysis output).
    Aligns depths to video timestamps using earliest image timestamp as t0.
    """

    pothole_prefix: str = "pothole_"

    def extract(
        self,
        *,
        segment_dir: Path,
        raw_images_dir: Path,
        output_path: Path,
    ) -> DepthTimeline:
        """Extract depth timeline and return result with path."""
        # Get earliest timestamp from raw_images
        image_timestamps = []
        if raw_images_dir.is_dir():
            for f in raw_images_dir.iterdir():
                if f.suffix.lower() == ".jpg":
                    match = re.match(r"(\d+\.\d+)", f.stem)
                    if match:
                        image_timestamps.append(float(match.group(1)))

        if not image_timestamps:
            raise ValueError(
                f"No timestamped images found in {raw_images_dir}")

        first_timestamp = min(image_timestamps)

        # Discover pothole directories
        pothole_dirs = sorted(
            p
            for p in segment_dir.iterdir()
            if p.is_dir() and p.name.startswith(self.pothole_prefix)
        )

        entries: List[DepthTimelineEntry] = []

        for pothole_dir in pothole_dirs:
            # Get pothole timestamp from image in folder
            pothole_ts: Optional[float] = None
            for f in pothole_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".png"):
                    match = re.match(r"(\d+\.\d+)", f.stem)
                    if match:
                        pothole_ts = float(match.group(1))
                        break

            if pothole_ts is None:
                logger.warning(
                    "No timestamped image in %s, skipping", pothole_dir)
                continue

            # Get depth from inspection_result.json
            depth = self._get_pothole_depth(pothole_dir)
            if depth is None:
                logger.warning("No depth found in %s, skipping", pothole_dir)
                continue

            # Calculate relative timestamp
            rel_seconds = round(pothole_ts - first_timestamp, 6)

            entries.append(
                DepthTimelineEntry(
                    video_timestamp=str(rel_seconds),
                    pothole_depth=depth,
                )
            )

        # Sort by timestamp
        entries.sort(key=lambda e: float(e.video_timestamp))

        # Write JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"pothole_depth": e.pothole_depth, "video_timestamp": e.video_timestamp}
            for e in entries
        ]
        output_path.write_text(json.dumps(data, indent=4), encoding="utf-8")

        logger.info("Wrote depth timeline: %s (%d entries)",
                    output_path, len(entries))

        return DepthTimeline(entries=entries, path=output_path)

    def _get_pothole_depth(self, pothole_dir: Path) -> Optional[float]:
        """Extract depth from inspection_result.json (paired_assets_analysis)."""
        inspection_json = pothole_dir / "inspection_result.json"
        try:
            data = json.loads(inspection_json.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = None
        except Exception as e:
            logger.warning(
                "Failed to parse inspection_result.json in %s: %s", pothole_dir, e)
            data = None

        if data is not None:
            try:
                metadata = data.get("metadata") if isinstance(
                    data, dict) else None
                if not isinstance(metadata, dict):
                    return None

                # Preferred shape for cluster folders: `metadata.metrics.max_depth`
                m = metadata.get("metrics")
                if isinstance(m, dict) and "max_depth" in m:
                    return float(m["max_depth"])

                # Multi-cluster shape: pick best depth among clusters
                clusters = metadata.get("clusters")
                if isinstance(clusters, list) and clusters:
                    best_depth = float("-inf")
                    for c in clusters:
                        if not isinstance(c, dict):
                            continue
                        cm = c.get("metrics")
                        if not isinstance(cm, dict):
                            continue
                        try:
                            d = float(cm.get("max_depth"))
                        except Exception:
                            continue
                        if d > best_depth:
                            best_depth = d
                    if best_depth != float("-inf"):
                        return float(best_depth)
            except Exception as e:
                logger.warning(
                    "Failed to extract depth from inspection_result.json in %s: %s", pothole_dir, e)

        return None


# ---------------------------------------------------------------------------
# GpsConverter adapter
# ---------------------------------------------------------------------------


@dataclass
class FilesystemGpsConverter:
    """Converts GPS text log to structured JSON."""

    def convert(
        self,
        *,
        gps_file_path: Path,
        output_path: Path,
    ) -> GpsSeries:
        """Convert GPS file and return series with path."""
        entries: List[GpsEntry] = []

        with gps_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        ts = float(parts[0])
                        lat = float(parts[1])
                        lng = float(parts[2])
                        alt = float(parts[3]) if len(parts) >= 4 else None
                        entries.append(
                            GpsEntry(timestamp=ts, lat=lat, lng=lng, alt=alt))
                    except ValueError:
                        continue

        # Sort by timestamp
        entries.sort(key=lambda e: e.timestamp)

        # Write JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "timestamp": e.timestamp,
                "lat": e.lat,
                "lng": e.lng,
                "alt": e.alt,
            }
            for e in entries
        ]
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        logger.info("Wrote GPS series: %s (%d entries)",
                    output_path, len(entries))

        return GpsSeries(entries=entries, path=output_path)


# ---------------------------------------------------------------------------
# SegmentArtifactsRepository adapter
# ---------------------------------------------------------------------------


@dataclass
class FilesystemSegmentArtifactsRepository:
    """Persists segment processing results to filesystem.

    This implementation is a no-op beyond what the individual ports already
    write. It can be extended to:
    - Copy artifacts to a target tree structure
    - Write a consolidated manifest
    - Upload to cloud storage
    """

    def save(self, *, result: SegmentProcessingResult) -> None:
        """Persist the processing result.

        Currently all artifacts are already written by their respective ports.
        This method can be extended for additional persistence logic.
        """
        # All artifacts are already persisted by their ports.
        # Future: copy to target tree, write manifest, etc.
        logger.debug(
            "SegmentArtifactsRepository.save() called for segment: %s (status=%s)",
            result.segment.id.value,
            result.status.value,
        )
