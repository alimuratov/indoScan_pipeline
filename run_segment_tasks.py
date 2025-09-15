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
    script_path = os.path.join(os.path.dirname(__file__), "process_gps.py")
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
    script_path = os.path.join(os.path.dirname(__file__), "retrieve_depth_timestamps.py")
    image_folder = os.path.join(segment_dir, images_folder_name)
    output_json = os.path.join(segment_dir, "aggregated_depth_data.json")

    if not os.path.isfile(script_path):
        logging.error("retrieve_depth_timestamps.py not found: %s", script_path)
        return
    if not os.path.isdir(image_folder):
        logging.warning("Image folder not found for segment: %s", image_folder)
        return

    rc = run_subprocess([
        sys.executable,
        script_path,
        "--image-folder", image_folder,
        "--root-dir", segment_dir,
        "--output", output_json,
    ])
    if rc == 0:
        logging.info("retrieve_depth_timestamps completed: %s", output_json)


def run_create_video(segment_dir: str, images_folder_name: str, fps: int) -> None:
    # Import and call function directly to control input/output
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from create_video import create_video_from_images  # type: ignore
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traverse segments and run GPS/depth/video tasks")
    parser.add_argument("--root", required=True, help="Root directory of roads or a single segment directory")
    parser.add_argument("--images-folder-name", default="raw_images", help="Name of images folder inside each segment")
    parser.add_argument("--fps", type=int, default=10, help="FPS for created videos")
    parser.add_argument("--odometry-filename", default="imu.txt", help="Filename of odometry file in each segment (timestamp x y z ...)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        run_process_gps(segment)
        run_retrieve_depth(segment, args.images_folder_name)
        # run_create_video(segment, args.images_folder_name, args.fps)
        # Route length calculation (optional per segment)
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            from calculate_road_length import calculate_route_length  # type: ignore
            odo_path = os.path.join(segment, args.odometry_filename)
            if os.path.isfile(odo_path):
                length_m = calculate_route_length(odo_path)
                length_km = length_m / 1000.0
                logging.info("Route length for segment %s: %.2f m (%.5f km)", segment, length_m, length_km)
                # write route_length.json
                out_json = os.path.join(segment, "route_length.json")
                try:
                    with open(out_json, "w") as f:
                        f.write("{\n  \"meters\": %.6f,\n  \"kilometers\": %.6f\n}\n" % (length_m, length_km))
                    logging.info("Wrote %s", out_json)
                except Exception:
                    logging.exception("Failed to write route_length.json for segment: %s", segment)
            else:
                logging.info("Odometry file not found, skipping length calc: %s", odo_path)
        except Exception:
            logging.exception("Route length calculation failed for segment: %s", segment)

    logging.info("All segments processed: %d", len(segments))


if __name__ == "__main__":
    main()


