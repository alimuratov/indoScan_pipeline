"""run_segment_tasks

Traverse segment folders and run:
- process_gps.py (if present) to generate segment-level JSON from gps.txt
- retrieve_depth_timestamps.py to generate aggregated_depth_data.json
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
import subprocess
import sys
from typing import List

# Ensure the scripts root is importable (so we can import sensors/media packages)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)


def list_immediate_subdirs(parent_dir: str) -> List[str]:
    try:
        return [
            os.path.join(parent_dir, name)
            for name in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, name))
        ]
    except Exception:
        return []


def is_pothole_dir(path: str) -> bool:
    try:
        names = set(os.listdir(path))
    except Exception:
        return False
    # Consider it a pothole dir if it contains at least one of these artifacts
    has_artifact = any(
        any(name.lower().endswith(ext) for ext in (".jpg", ".png", ".pcd"))
        for name in names
    ) or ("output.txt" in names)
    return has_artifact


def is_segment_dir(path: str, images_folder_name: str) -> bool:
    # Heuristic: has raw_images folder OR has at least one pothole-like subdir
    if os.path.isdir(os.path.join(path, images_folder_name)):
        return True
    try:
        for child in list_immediate_subdirs(path):
            if is_pothole_dir(child):
                return True
    except Exception:
        return False
    return False


def discover_segments(root: str, images_folder_name: str) -> List[str]:
    # If root itself looks like a segment, return it
    if is_segment_dir(root, images_folder_name):
        logging.info("Using single segment root: %s", root)
        return [root]

    segments: List[str] = []
    for maybe_road in list_immediate_subdirs(root):
        for maybe_segment in list_immediate_subdirs(maybe_road):
            if is_segment_dir(maybe_segment, images_folder_name):
                logging.info("Detected segment: %s", maybe_segment)
                segments.append(maybe_segment)
            else:
                logging.debug("Skipping non-segment dir: %s", maybe_segment)

    if not segments:
        logging.warning("No segments detected under: %s", root)
    return segments


def run_subprocess(cmd: List[str], cwd: str | None = None) -> int:
    logging.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd or os.getcwd())
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if proc.stdout:
            logging.debug("stdout: %s", proc.stdout.strip())
        if proc.stderr:
            logging.debug("stderr: %s", proc.stderr.strip())
        if proc.returncode != 0:
            logging.warning("Command failed (%d): %s", proc.returncode, " ".join(cmd))
        return proc.returncode
    except FileNotFoundError:
        logging.error("Executable not found for command: %s", cmd[0])
        return 127
    except Exception as exc:
        logging.exception("Subprocess error for: %s", " ".join(cmd))
        return 1


def run_process_gps(segment_dir: str) -> None:
    script_path = os.path.abspath(os.path.join(_SCRIPTS_ROOT, "sensors", "process_gps.py"))
    gps_txt = os.path.join(segment_dir, "gps.txt")
    output_json = os.path.join(segment_dir, "imu.json")

    if not os.path.isfile(script_path):
        logging.info("process_gps.py not found, skipping for segment: %s", segment_dir)
        return
    if not os.path.isfile(gps_txt):
        logging.warning("gps.txt not found in segment: %s", segment_dir)
        return

    rc = run_subprocess([sys.executable, script_path, gps_txt, output_json])
    if rc == 0:
        logging.info("process_gps completed: %s -> %s", gps_txt, output_json)


def run_retrieve_depth(segment_dir: str, images_folder_name: str) -> None:
    script_path = os.path.abspath(os.path.join(_SCRIPTS_ROOT, "sensors", "segment_depth_timestamps.py"))
    image_folder = os.path.join(segment_dir, images_folder_name)
    output_json = os.path.join(segment_dir, "segment_depth_timestamps.json")

    if not os.path.isfile(script_path):
        logging.error("retrieve_depth_timestamps.py not found: %s", script_path)
        return
    if not os.path.isdir(image_folder):
        logging.warning("Image folder not found for segment: %s", image_folder)
        return

    rc = run_subprocess([
        sys.executable,
        script_path,
        "--root-dir", segment_dir,
        "--images-folder-name", images_folder_name,
        "--output", output_json,
    ])
    if rc == 0:
        logging.info("retrieve_depth_timestamps completed: %s", output_json)


def run_create_video(segment_dir: str, images_folder_name: str, fps: int) -> None:
    # Import and call function directly to control input/output
    try:
        from media.create_video import create_video_from_images  # type: ignore
    except Exception:
        logging.exception("Failed to import create_video module")
        return

    input_folder = os.path.join(segment_dir, images_folder_name)
    output_path = os.path.join(segment_dir, "output_video.mp4")
    if not os.path.isdir(input_folder):
        logging.warning("raw_images folder not found for segment: %s", input_folder)
        return

    try:
        logging.info("Creating video for segment: %s", segment_dir)
        create_video_from_images(input_folder, output_path, fps)
    except Exception:
        logging.exception("create_video failed for segment: %s", segment_dir)

def run_calculate_route_length(segment_dir: str, odometry_filename: str) -> None:
    odo_path = os.path.join(segment_dir, odometry_filename)
    from sensors.calculate_road_length import calculate_route_length 
    try: 
        length_m = calculate_route_length(odo_path)
        length_km = length_m / 1000.0
    except Exception:
        logging.exception("Failed to calculate route length for segment: %s", segment_dir)
        return

    logging.info("Route length for segment %s: %.2f m (%.5f km)", segment_dir, length_m, length_km)
    out_json = os.path.join(segment_dir, "route_length.json")
    try:
        with open(out_json, "w") as f:
            f.write("{\n  \"meters\": %.6f,\n  \"kilometers\": %.6f\n}\n" % (length_m, length_km))
        logging.info("Wrote %s", out_json)
    except Exception:
        logging.exception("Failed to write route_length.json for segment: %s", segment_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traverse segments and run GPS/depth/video tasks")
    parser.add_argument("--root", required=True, help="Root directory of roads or a single segment directory")
    parser.add_argument("--images-folder-name", default="raw_images", help="Name of images folder inside each segment")
    parser.add_argument("--fps", type=int, default=10, help="FPS for created videos")
    parser.add_argument("--odometry-filename", default="imu.txt", help="Filename of odometry file in each segment (timestamp x y z ...)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser


def main() -> None:
    p = build_parser()
    cfg_path = p.parse_known_args()[0].config

    from common.config import load_config
    cfg = load_config(cfg_path)

    p.set_defaults(
        root=cfg.paths.source_roads_root,
        images_folder_name=cfg.paths.images_folder_name,
        odometry_filename=cfg.paths.odometry_filename,
        fps=cfg.media.fps,
        log_level=cfg.logging.level,
    )

    args = p.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        logging.error("Root is not a directory: %s", root)
        sys.exit(1)

    segments = discover_segments(root, args.images_folder_name)
    for segment in segments:
        logging.info("\n=== Processing segment: %s ===", segment)
        
        run_estimate_depth_for_potholes(segment)
        run_process_gps(segment)
        run_retrieve_depth(segment, args.images_folder_name)
        run_create_video(segment, args.images_folder_name, args.fps)
        run_calculate_route_length(segment, args.odometry_filename)
        

    logging.info("All segments processed: %d", len(segments))


if __name__ == "__main__":
    main()


