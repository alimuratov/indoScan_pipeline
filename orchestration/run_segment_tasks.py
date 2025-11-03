"""run_segment_tasks

Traverse segment folders and run:
- process_gps.py (if present) to generate segment-level JSON from gps.txt
- segment_depth_timestamps.py to generate segment_depth_timestamps.json
- create_video.py to export a video from raw_images

Assumptions:
- Directory structure: roads/<road_id>/<segment_id>/<pothole_id>
- Each segment contains a 'raw_images' folder with timestamped images
- gps.txt resides in the segment folder (same level as raw_images and potholes)

Usage:
    python scripts/run_segment_tasks.py \
        --root /path/to/roads \
        --images-folder-name raw_images \
        --fps 10 \
        --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
import os
from common.discovery import discover_segments
from common.cli import add_config_arg, add_log_level_arg, setup_logging, parse_args_with_config
from processing.run_estimate_depth_for_potholes import run_estimate_depth_for_potholes
from validation.validate_segment_tasks import validate_source_roads_root_before_processing, validate_source_roads_root_after_processing
from validation.validation_helpers import log_issues
from common.logging import CountingHandler
from exceptions.exceptions import StepPreconditionError
from media.create_video import create_video_from_images 

def process_segment(segment: str, args) -> tuple[int, int]:
    counter = CountingHandler()
    logging.getLogger().addHandler(counter)

    seen_issues = set()

    try:

        run_estimate_depth_for_potholes(segment, os.path.join(args.scripts_root, args.estimate_script), args.config)
        run_process_gps(segment, args.gps_filename)
        run_retrieve_depth(segment, args.images_folder_name)
        run_create_video(segment, args.images_folder_name, args.fps)
        run_calculate_route_length(segment, args.imu_filename) 

    except StepPreconditionError as e:
        key = e.code
        if key not in seen_issues:
            seen_issues.add(key)
            logging.error("Segment: %s, %s: %s", os.path.basename(segment), key, str(e))
    
    logging.getLogger().removeHandler(counter)

    return counter.warnings, counter.errors

def run_process_gps(segment_dir: str, process_gps_input_filename: str) -> None:
    gps_txt = os.path.join(segment_dir, process_gps_input_filename)
    output_json = os.path.join(segment_dir, "imu.json")

    from sensors.process_gps import process_imu_file
    process_imu_file(gps_txt, output_json)

    logging.debug("process_gps completed: %s -> %s", gps_txt, output_json)

def run_retrieve_depth(segment_dir: str, images_folder_name: str) -> None:
    image_folder = os.path.join(segment_dir, images_folder_name)
    output_json = os.path.join(segment_dir, "segment_depth_timestamps.json")

    from sensors.segment_depth_timestamps import process_segment_folder, write_depth_timestamps
    entries = process_segment_folder(segment_dir, images_folder_name)
    write_depth_timestamps(entries, output_json)

def run_create_video(segment_dir: str, images_folder_name: str, fps: int) -> None:
    input_folder = os.path.join(segment_dir, images_folder_name)
    output_path = os.path.join(segment_dir, "output_video.mp4")
   
    try:
        logging.info("Creating video for segment: %s", segment_dir)
        create_video_from_images(input_folder, output_path, fps)
    except Exception:
        raise StepPreconditionError(
            "CREATE_VIDEO_FAILED",
            f"create_video failed for segment: {segment_dir}",
            context="run_create_video",
        )

def run_calculate_route_length(segment_dir: str, odometry_filename: str) -> None:
    odo_path = os.path.join(segment_dir, odometry_filename)
    from sensors.calculate_road_length import calculate_route_length 
    try: 
        length_m = calculate_route_length(odo_path)
        length_km = length_m / 1000.0
    except Exception as e:
        raise StepPreconditionError(
            "ROUTE_LENGTH_CALCULATION_FAILED",
            f"Failed to calculate route length for segment: {segment_dir}",
            context="run_calculate_route_length",
        ) from e

    logging.debug("Route length for segment %s: %.2f m (%.5f km)", segment_dir, length_m, length_km)
    out_json = os.path.join(segment_dir, "route_length.json")
    try:
        with open(out_json, "w") as f:
            f.write("{\n  \"meters\": %.6f,\n  \"kilometers\": %.6f\n}\n" % (length_m, length_km))
        logging.debug("Wrote %s", out_json)
    except Exception:
        raise StepPreconditionError(
            "ROUTE_LENGTH_WRITE_FAILED",
            f"Failed to write route_length.json for segment: {segment_dir}",
            context="run_calculate_route_length",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traverse segments and run GPS/depth/video tasks")
    add_config_arg(parser); add_log_level_arg(parser)
    parser.add_argument("--root", help="Root directory of roads or a single segment directory")
    parser.add_argument("--images-folder-name", default="raw_images", help="Name of images folder inside each segment")
    parser.add_argument("--fps", type=int, default=10, help="FPS for created videos")
    parser.add_argument("--workspace-path", help="Path to scripts")
    parser.add_argument("--estimate-script", help="Path to estimate script")
    parser.add_argument("--scripts-root", help="Path to scripts root")
    parser.add_argument("--imu-filename", help="Filename of imu file in each segment (timestamp x y z ...)")
    parser.add_argument("--gps-filename", help="Filename of gps file in each segment (timestamp lat lng ...)")
    return parser


def main() -> None:
    args, cfg = parse_args_with_config(
        build_parser, 
        lambda cfg: {
            "workspace_root": cfg.paths.workspace_root,
            "source_roads_root": cfg.paths.source_roads_root,
            "images_folder_name": cfg.paths.images_folder_name,
            "fps": cfg.media.fps,
            "log_level": cfg.logging.level,
            "estimate_script": cfg.paths.estimate_script,
            "gps_filename": cfg.paths.gps_filename,
            "imu_filename": cfg.paths.imu_filename,
            "scripts_root": cfg.paths.scripts_root,
        }
    )

    setup_logging(args.log_level)

    issues = validate_source_roads_root_before_processing(cfg)

    errors = log_issues(issues, "error")

    if errors:
        logging.info("❌ Fix errors before running segment tasks")
        return 

    source_roads_root_path = os.path.abspath(os.path.join(args.workspace_root, args.source_roads_root))

    logging.debug("Source roads root path: %s", source_roads_root_path)

    segments = discover_segments(source_roads_root_path)

    logging.debug("Found %d segments to process", len(segments))

    for segment in segments:
        warnings, errors = process_segment(segment, args)
        logging.info("Segment: %s, Warnings: %d, Errors: %d", os.path.basename(segment), warnings, errors)


    issues = validate_source_roads_root_after_processing(cfg)
    
    errors = log_issues(issues, "error")

    if errors:
        logging.info("❌ Re-run segment tasks")
        return 


if __name__ == "__main__":
    main()


