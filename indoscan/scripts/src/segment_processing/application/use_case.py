"""Application use case for segment processing.

Orchestrates the segment processing workflow:
1. Build workspace (resolve paths)
2. Run computation ports (video, IMU, route length, depth timeline, GPS)
3. Build segment metadata
4. Optionally persist results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from segment_processing.domain.model import (
    DepthTimeline,
    GpsSeries,
    RouteLengthResult,
    Segment,
    SegmentId,
    SegmentMeta,
    SegmentProcessingResult,
    SegmentProcessingStatus,
    VerticalDisplacementSeries,
    VideoArtifact,
)
from segment_processing.ports import (
    DepthTimelineExtractor,
    GpsConverter,
    ImuProcessor,
    RouteLengthCalculator,
    SegmentArtifactsRepository,
    SegmentWorkspace,
    SegmentWorkspaceFactory,
    VideoCreator,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessSegmentConfig:
    """Configuration for segment processing."""

    fps: int = 10
    imu_interval_seconds: float = 30.0
    persist_results: bool = True


class ProcessSegmentUseCase:
    """Application service that orchestrates segment processing.

    Typical flow:
    1. Build workspace from segment directory
    2. Create video from raw_images
    3. Process IMU to vertical displacement series
    4. Calculate route length from odometry
    5. Extract depth timeline from pothole inspections
    6. Convert GPS to structured JSON
    7. Build segment metadata (start/end location, length)
    8. Optionally persist all results
    """

    def __init__(
        self,
        *,
        workspace_factory: SegmentWorkspaceFactory,
        video_creator: VideoCreator,
        imu_processor: ImuProcessor,
        route_length_calculator: RouteLengthCalculator,
        depth_timeline_extractor: DepthTimelineExtractor,
        gps_converter: GpsConverter,
        artifacts_repository: Optional[SegmentArtifactsRepository] = None,
    ) -> None:
        self._workspace_factory = workspace_factory
        self._video_creator = video_creator
        self._imu_processor = imu_processor
        self._route_length_calculator = route_length_calculator
        self._depth_timeline_extractor = depth_timeline_extractor
        self._gps_converter = gps_converter
        self._artifacts_repository = artifacts_repository

    def run(
        self,
        *,
        segment_dir: Path,
        config: Optional[ProcessSegmentConfig] = None,
    ) -> SegmentProcessingResult:
        """Process a segment and return the result.

        Args:
            segment_dir: Path to the source segment directory.
            config: Processing configuration (defaults applied if None).

        Returns:
            SegmentProcessingResult containing all artifacts and status.
        """
        cfg = config or ProcessSegmentConfig()
        warnings: list[str] = []
        errors: list[str] = []

        # Build workspace
        workspace = self._workspace_factory.build(segment_dir=segment_dir)

        # Create segment entity
        segment = Segment(
            id=SegmentId(value=segment_dir.name),
            source_dir=segment_dir,
        )

        logger.info("Processing segment: %s", segment.id.value)

        # --- Run computation ports ---

        video: Optional[VideoArtifact] = None
        imu_series: Optional[VerticalDisplacementSeries] = None
        route_length: Optional[RouteLengthResult] = None
        depth_timeline: Optional[DepthTimeline] = None
        gps_series: Optional[GpsSeries] = None

        # 1. Create video
        video = self._try_create_video(workspace, cfg.fps, warnings, errors)

        # 2. Process IMU
        imu_series = self._try_process_imu(
            workspace, cfg.imu_interval_seconds, warnings, errors
        )

        # 3. Calculate route length
        route_length = self._try_calculate_route_length(
            workspace, warnings, errors)

        # 4. Extract depth timeline
        depth_timeline = self._try_extract_depth_timeline(
            workspace, warnings, errors)

        # 5. Convert GPS
        gps_series = self._try_convert_gps(workspace, warnings, errors)

        # 6. Build segment metadata
        segment_meta = self._build_segment_meta(
            workspace, gps_series, route_length, warnings, errors
        )

        # Determine status
        status = self._determine_status(
            video, imu_series, route_length, depth_timeline, gps_series, errors
        )

        result = SegmentProcessingResult(
            segment=segment,
            status=status,
            video=video,
            imu_series=imu_series,
            route_length=route_length,
            depth_timeline=depth_timeline,
            gps_series=gps_series,
            segment_meta=segment_meta,
            warnings=warnings,
            errors=errors,
        )

        # 7. Persist if configured
        if cfg.persist_results and self._artifacts_repository:
            try:
                self._artifacts_repository.save(result=result)
                logger.info("Persisted segment artifacts for: %s",
                            segment.id.value)
            except Exception as e:
                logger.error("Failed to persist artifacts: %s", e)
                errors.append(f"Persistence failed: {e}")

        return result

    def _try_create_video(
        self,
        workspace: SegmentWorkspace,
        fps: int,
        warnings: list[str],
        errors: list[str],
    ) -> Optional[VideoArtifact]:
        """Attempt to create video, handling errors gracefully."""
        if not workspace.raw_images_dir.is_dir():
            warnings.append(
                f"raw_images not found: {workspace.raw_images_dir}")
            return None
        try:
            return self._video_creator.create_video(
                images_dir=workspace.raw_images_dir,
                output_path=workspace.output_video_path,
                fps=fps,
            )
        except Exception as e:
            errors.append(f"Video creation failed: {e}")
            logger.error("Video creation failed: %s", e)
            return None

    def _try_process_imu(
        self,
        workspace: SegmentWorkspace,
        interval: float,
        warnings: list[str],
        errors: list[str],
    ) -> Optional[VerticalDisplacementSeries]:
        """Attempt to process IMU, handling errors gracefully."""
        if not workspace.imu_txt.is_file():
            warnings.append(f"imu.txt not found: {workspace.imu_txt}")
            return None
        try:
            return self._imu_processor.create_series(
                imu_file_path=workspace.imu_txt,
                output_path=workspace.imu_json_path,
                interval_seconds=interval,
            )
        except Exception as e:
            errors.append(f"IMU processing failed: {e}")
            logger.error("IMU processing failed: %s", e)
            return None

    def _try_calculate_route_length(
        self,
        workspace: SegmentWorkspace,
        warnings: list[str],
        errors: list[str],
    ) -> Optional[RouteLengthResult]:
        """Attempt to calculate route length, handling errors gracefully."""
        # Route length uses imu.txt (odometry) in legacy
        if not workspace.imu_txt.is_file():
            warnings.append(
                f"imu.txt not found for route length: {workspace.imu_txt}")
            return None
        try:
            return self._route_length_calculator.calculate(
                odometry_file_path=workspace.imu_txt,
                output_path=workspace.route_length_json_path,
            )
        except Exception as e:
            errors.append(f"Route length calculation failed: {e}")
            logger.error("Route length calculation failed: %s", e)
            return None

    def _try_extract_depth_timeline(
        self,
        workspace: SegmentWorkspace,
        warnings: list[str],
        errors: list[str],
    ) -> Optional[DepthTimeline]:
        """Attempt to extract depth timeline, handling errors gracefully."""
        if not workspace.raw_images_dir.is_dir():
            warnings.append(
                f"raw_images not found for depth timeline: {workspace.raw_images_dir}"
            )
            return None
        try:
            return self._depth_timeline_extractor.extract(
                segment_dir=workspace.segment_dir,
                raw_images_dir=workspace.raw_images_dir,
                output_path=workspace.depth_timeline_json_path,
            )
        except Exception as e:
            errors.append(f"Depth timeline extraction failed: {e}")
            logger.error("Depth timeline extraction failed: %s", e)
            return None

    def _try_convert_gps(
        self,
        workspace: SegmentWorkspace,
        warnings: list[str],
        errors: list[str],
    ) -> Optional[GpsSeries]:
        """Attempt to convert GPS, handling errors gracefully."""
        if not workspace.gps_txt.is_file():
            warnings.append(f"gps.txt not found: {workspace.gps_txt}")
            return None
        try:
            return self._gps_converter.convert(
                gps_file_path=workspace.gps_txt,
                output_path=workspace.gps_json_path,
            )
        except Exception as e:
            errors.append(f"GPS conversion failed: {e}")
            logger.error("GPS conversion failed: %s", e)
            return None

    def _build_segment_meta(
        self,
        workspace: SegmentWorkspace,
        gps_series: Optional[GpsSeries],
        route_length: Optional[RouteLengthResult],
        warnings: list[str],
        errors: list[str],
    ) -> Optional[SegmentMeta]:
        """Build segment metadata from GPS and route length."""
        start_loc = ""
        end_loc = ""
        length_km = 0.0

        if gps_series:
            start_loc = gps_series.start_location
            end_loc = gps_series.end_location

        if route_length:
            length_km = route_length.kilometers

        meta = SegmentMeta(
            start_loc=start_loc,
            end_loc=end_loc,
            length_in_km=length_km,
            path=workspace.segment_meta_json_path,
        )

        # Write segment_meta.json
        try:
            import json

            workspace.segment_meta_json_path.parent.mkdir(
                parents=True, exist_ok=True)
            workspace.segment_meta_json_path.write_text(
                json.dumps(
                    {
                        "start_loc": meta.start_loc,
                        "end_loc": meta.end_loc,
                        "length_in_km": meta.length_in_km,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            errors.append(f"Failed to write segment_meta.json: {e}")
            logger.error("Failed to write segment_meta.json: %s", e)

        return meta

    def _determine_status(
        self,
        video: Optional[VideoArtifact],
        imu_series: Optional[VerticalDisplacementSeries],
        route_length: Optional[RouteLengthResult],
        depth_timeline: Optional[DepthTimeline],
        gps_series: Optional[GpsSeries],
        errors: list[str],
    ) -> SegmentProcessingStatus:
        """Determine overall processing status."""
        if errors:
            # If there are errors but some artifacts exist, it's partial
            artifacts = [video, imu_series,
                         route_length, depth_timeline, gps_series]
            if any(a is not None for a in artifacts):
                return SegmentProcessingStatus.PARTIAL
            return SegmentProcessingStatus.FAILED

        # All critical artifacts should exist for OK
        if video and imu_series and route_length and depth_timeline:
            return SegmentProcessingStatus.OK

        return SegmentProcessingStatus.PARTIAL
