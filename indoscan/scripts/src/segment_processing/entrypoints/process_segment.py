"""CLI entrypoint for segment processing.

Usage:
    python -m segment_processing.entrypoints.process_segment \
        --segment-dir /path/to/segment \
        --fps 10 \
        --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from segment_processing.application.use_case import (
    ProcessSegmentConfig,
    ProcessSegmentUseCase,
)
from segment_processing.domain.model import SegmentProcessingResult, SegmentProcessingStatus
from segment_processing.infrastructure.adapters import (
    FilesystemGpsConverter,
    LegacyDepthTimelineExtractor,
    LegacyImuProcessor,
    LegacyRouteLengthCalculator,
    OpenCVVideoCreator,
)
from segment_processing.infrastructure.data2_export import (
    Data2SegmentArtifactsRepository,
    Data2ConsolidatedJsonBuilder,
)
from segment_processing.infrastructure.workspace import (
    FilesystemSegmentWorkspaceFactory,
)


def _setup_logging(log_level: str) -> None:
    """Setup logging without pulling optional repo config deps (e.g. PyYAML)."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _print_result_summary(result: SegmentProcessingResult) -> None:
    """Print a human-friendly summary of the segment processing result."""
    print(f"Segment: {result.segment.id.value}")
    print(f"Status: {result.status.value}")

    if result.video:
        print(
            f"  Video: {result.video.path} ({result.video.frame_count} frames)")
    if result.imu_series:
        print(
            f"  IMU series: {result.imu_series.path} ({len(result.imu_series.entries)} entries)")
    if result.route_length:
        print(
            f"  Route length: {result.route_length.meters:.2f} m "
            f"({result.route_length.kilometers:.4f} km)"
        )
    if result.depth_timeline:
        print(
            f"  Depth timeline: {result.depth_timeline.path} ({len(result.depth_timeline.entries)} entries)")
    if result.gps_series:
        print(
            f"  GPS series: {result.gps_series.path} ({len(result.gps_series.entries)} entries)")
    if result.segment_meta:
        print(f"  Segment meta: {result.segment_meta.path}")

    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")

    if result.errors:
        print("Errors:")
        for e in result.errors:
            print(f"  - {e}")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Process a segment: create video, extract IMU, calculate route, etc."
    )
    parser.add_argument(
        "--segment-dir",
        type=Path,
        required=True,
        help="Path to the segment directory to process.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS for output video (default: 10).",
    )
    parser.add_argument(
        "--imu-interval",
        type=float,
        default=30.0,
        help="IMU sampling interval in seconds (default: 30).",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not persist results (dry run).",
    )
    parser.add_argument(
        "--export-data2-root",
        type=Path,
        default=None,
        help=(
            "Optional: export a legacy-compatible target tree under this folder "
            "(e.g. indoScan/output/data2). Creates/updates copy_manifest.json and Roads/."
        ),
    )
    parser.add_argument(
        "--build-json",
        type=Path,
        default=None,
        metavar="OUTPUT_PATH",
        help=(
            "Optional: build consolidated roads JSON (like trial_output.json) "
            "from the Data2 tree. Requires --export-data2-root."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser


def process_segment(
    *,
    segment_dir: Path,
    fps: int = 10,
    imu_interval: float = 30.0,
    persist: bool = True,
    export_data2_root: Path | None = None,
    build_json_output: Path | None = None,
) -> int:
    """Process a single segment and return exit code.

    This is the composition root: wires up all dependencies.

    Args:
        segment_dir: Path to the segment directory to process.
        fps: FPS for output video.
        imu_interval: IMU sampling interval in seconds.
        persist: Whether to persist results.
        export_data2_root: Optional path to export Data2 tree.
        build_json_output: Optional path to write consolidated roads JSON
            (requires export_data2_root).

    Returns:
        0 on success, 1 on failure.
    """
    # Wire up infrastructure
    workspace_factory = FilesystemSegmentWorkspaceFactory()
    video_creator = OpenCVVideoCreator()
    imu_processor = LegacyImuProcessor()
    route_calculator = LegacyRouteLengthCalculator()
    depth_extractor = LegacyDepthTimelineExtractor()
    gps_converter = FilesystemGpsConverter()
    artifacts_repo = Data2SegmentArtifactsRepository(
        export_root=export_data2_root)

    # Create use case
    use_case = ProcessSegmentUseCase(
        workspace_factory=workspace_factory,
        video_creator=video_creator,
        imu_processor=imu_processor,
        route_length_calculator=route_calculator,
        depth_timeline_extractor=depth_extractor,
        gps_converter=gps_converter,
        artifacts_repository=artifacts_repo,
    )

    # Configure
    config = ProcessSegmentConfig(
        fps=fps,
        imu_interval_seconds=imu_interval,
        persist_results=persist,
    )

    # Run
    result = use_case.run(segment_dir=segment_dir, config=config)

    _print_result_summary(result)

    # Build consolidated JSON if requested
    if build_json_output is not None and export_data2_root is not None:
        json_builder = Data2ConsolidatedJsonBuilder()
        target_roads_dir = export_data2_root / "Roads"
        json_builder.build(
            target_roads_dir=target_roads_dir,
            output_path=build_json_output,
        )
        print(f"Built consolidated JSON: {build_json_output}")

    return 0 if result.status == SegmentProcessingStatus.OK else 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)

    # Validate: --build-json requires --export-data2-root
    if args.build_json and not args.export_data2_root:
        parser.error("--build-json requires --export-data2-root")

    return process_segment(
        segment_dir=args.segment_dir.resolve(),
        fps=args.fps,
        imu_interval=args.imu_interval,
        persist=not args.no_persist,
        export_data2_root=args.export_data2_root.resolve(
        ) if args.export_data2_root else None,
        build_json_output=args.build_json.resolve(
        ) if args.build_json else None,
    )


if __name__ == "__main__":
    sys.exit(main())
