"""build_road_json

Generate a consolidated road-scanning JSON from a folder hierarchy of
roads → segments → potholes. The script also tries to integrate outputs from
the existing tools:
- estimate_depth.py → per-pothole output.txt (depth/volume lines)
- process_imu.py → imu.json entries (segment-level or collected from pothole dirs)
- segment_depth_timestamps.py → segment_depth_timestamps.json (segment-level)

Assumptions and behavior:
- Root directory contains multiple road folders. Each road folder contains one
  or more segment folders. Each segment folder contains one or more pothole
  folders. Each pothole folder contains: an image (.jpg/.png), a lidar scan
  (.pcd), a gps.txt, and an output.txt produced by estimate_depth.py.
- start_loc and end_loc for a segment are derived from GPS logs found in all
  pothole gps.txt files within that segment. We take the earliest and latest
  timestamps across those files to form "lat, lng" strings.
- gps_data: If an imu.json file exists in the segment (or nested within any
  pothole), it will be loaded. If multiple are found, they are merged and
  sorted by video_timestamp (as float). If none are found, gps_data is empty.
- depth_data: If segment_depth_timestamps.json (or legacy aggregated_depth_data.json) is present
  under the segment, we load it. Otherwise, we estimate it by parsing pothole
  image timestamps and depths from output.txt, computing a per-segment relative
  time using the
  earliest pothole image timestamp as t0.
- Paths in the output JSON are made relative to the output JSON's directory.

Usage:
    python scripts/build_road_json.py \
        --roads-root /path/to/roads_root \
        --output /path/to/road-scanning-POC-json.json

Where roads_root has a structure like:
    roads_root/
      road-1/
        segment-001/
          pothole-001/
            gps.txt
            output.txt
            1756371511.4001415.jpg
            1756371511.4001415.pcd
          pothole-002/ ...
        segment-002/ ...
      road-2/ ...

This script is resilient to naming variance (e.g., "RD-1"/"RS-1"/"PT-1"). It
detects pothole folders as leaf directories under a segment. A segment is any
directory under a road that has at least one pothole-like subdirectory.

Output path policy:
- Provide paths relative to the target Roads directory (not the source tree and
  not relative to the JSON file). Use --target-roads-dir to set that root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
import shutil
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Reuse common helpers
from common.fs import list_immediate_subdirs, find_files_with_extensions, find_immediate_files_with_extensions, copy_asset, ensure_parent_dir, find_first_matching_file
from common.discovery import discover_target_roads
# target-only build; GPS utilities not required


# ----------------------------- Utility helpers ----------------------------- #


def generate_prefixed_uuid(prefix: str) -> str:
    """Generate a random id string with a given prefix, e.g., "rd-<uuid>".

    Interaction: Used by road/segment/pothole builders to assign stable IDs
    referenced across the JSON structure.
    """
    return f"{prefix}-{uuid.uuid4()}"

def parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def read_text_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return f.readlines()


def to_relpath(path: str, relative_to_dir: str) -> str:
    try:
        return os.path.relpath(path, start=relative_to_dir)
    except Exception:
        logging.debug("to_relpath fallback for path=%s relative_to=%s", path, relative_to_dir)
        return path


def join_target(*parts: str) -> str:
    return os.path.join(*parts)
# (copy-only API moved to copy_roads_to_target.py)


# ----------------------------- Depth/volume parse -------------------------- #


def parse_depth_and_volume_from_output(output_txt_path: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse depth and volume from output.txt.

    Tries the following patterns (in order):
    - Depth: "Surface-based max depth: <float> m" or "Max depth: <float> m"
    - Volume: "Surface-based volume: <float> m" or "Volume: <float> m"
    Returns (depth, volume) as floats if found, else None for missing values.
    """
    depth: Optional[float] = None
    volume: Optional[float] = None
    try:
        for line in read_text_lines(output_txt_path):
            low = line.lower()
            if depth is None:
                if "surface-based max depth" in low:
                    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
                    if nums:
                        depth = parse_float(nums[0])
            if volume is None:
                if "surface-based volume" in low:
                    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
                    if nums:
                        volume = parse_float(nums[0])
    except FileNotFoundError:
        logging.debug("output.txt not found: %s", output_txt_path)
    return depth, volume


# --------------------------------- IMU load -------------------------------- #


def load_imu_entries_from_segment(segment_dir: str) -> List[Dict]:
    """Find and merge any imu.json files under a segment directory."""
    imu_files: List[str] = []
    for dirpath, _, filenames in os.walk(segment_dir):
        for fname in filenames:
            if fname.lower() == "imu.json":
                imu_files.append(os.path.join(dirpath, fname))
    logging.debug("Found imu.json files: %s", imu_files)
    merged: List[Dict] = []
    for path in imu_files:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged.extend(data)
        except Exception:
            from exceptions.exceptions import StepPreconditionError
            raise StepPreconditionError(
                "IMU_JSON_READ_FAILED",
                f"Failed to read imu.json: {path}",
                context="build_road_json.load_imu_entries_from_segment",
            )
    # Normalize and sort unique by video_timestamp (string) cast to float
    seen_ts = set()
    normalized: List[Dict] = []
    for item in merged:
        ts = str(item.get("video_timestamp", ""))
        val = item.get("vertical_displacement")
        if ts in seen_ts:
            continue
        if isinstance(val, (int, float)):
            normalized.append({
                "vertical_displacement": float(val),
                "video_timestamp": ts,
            })
            seen_ts.add(ts)
    normalized.sort(key=lambda d: float(d["video_timestamp"]))
    return normalized


# ------------------------ Depth data aggregate (segment) ------------------- #


def load_segment_depth_data(segment_dir: str) -> Optional[List[Dict]]:
    """Load per-segment depth/timestamp data from a single JSON if present.

    Looks only for one of these files in the segment directory tree (first match wins):
    - segment_depth_timestamps.json (preferred)
    - aggregated_depth_data.json (legacy fallback)

    Returns:
        Optional[List[Dict]]: the list as-is (assuming it's already sorted), or None if not found/invalid.
    """
    for dirpath, _, filenames in os.walk(segment_dir):
        for fname in filenames:
            if fname.lower() in ("segment_depth_timestamps.json", "aggregated_depth_data.json"):
                try:
                    with open(os.path.join(dirpath, fname), "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            return data
                except Exception:
                    from exceptions.exceptions import StepPreconditionError
                    raise StepPreconditionError(
                        "DEPTH_TIMESTAMPS_JSON_READ_FAILED",
                        f"Failed to load depth timestamps JSON in {dirpath}",
                        context="build_road_json.load_segment_depth_data",
                    )
    return None


# --------------------------- Pothole parsing (segment) --------------------- #


def parse_potholes(segment_dir: str, road_segment_id: str, target_roads_dir: str, road_id: str, segment_id: str) -> List[Dict]:
    """Build pothole array for a segment by reading assets from the target tree.

    Interaction: Called by segment builder to populate the "potholes" array.
    It copies image/pcd assets to the target tree and maps GPS to nearest
    image timestamp when available.
    """
    # Determine layout: source (pothole dirs at segment root) or target (Potholes/<id>)
    target_potholes_root = os.path.join(segment_dir, "Potholes")
    pothole_dirs = [
        p for p in list_immediate_subdirs(target_potholes_root)
        if os.path.isdir(p)
    ]

    logging.debug("Pothole directories under %s: %s", segment_dir, pothole_dirs)
    # Load segment GPS list from JSON (produced during copy phase)
    segment_gps_data: List[Dict] = []
    seg_gps_json = os.path.join(segment_dir, "segment_gps.json")
    if os.path.isfile(seg_gps_json):
        try:
            with open(seg_gps_json, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    segment_gps_data = data
        except Exception:
            from exceptions.exceptions import StepPreconditionError
            raise StepPreconditionError(
                "SEGMENT_GPS_JSON_READ_FAILED",
                f"Failed to read {seg_gps_json}",
                context="build_road_json.build_segment_payload",
            )
    potholes: List[Dict] = []

    for pothole_dir in pothole_dirs:
        pothole_id = os.path.basename(pothole_dir)

        # Image path
        image_path = None
        lidar_path = None
        image_folder = os.path.join(pothole_dir, "Image")
        if os.path.isdir(image_folder):
            imgs = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))]
            if imgs:
                image_path = imgs[0]
        lidar_folder = os.path.join(pothole_dir, "Lidar Scan")
        if os.path.isdir(lidar_folder):
            pcds = [os.path.join(lidar_folder, f) for f in os.listdir(lidar_folder) if f.lower().endswith(".pcd")]
            if pcds:
                lidar_path = pcds[0]

        image_rel = ""
        lidar_rel = ""
        if image_path and image_path.startswith(target_roads_dir):
            image_rel = to_relpath(image_path, target_roads_dir)
        if lidar_path and lidar_path.startswith(target_roads_dir):
            lidar_rel = to_relpath(lidar_path, target_roads_dir)

        # Depth/volume/area from pothole_meta.json (written during copy phase)
        depth = None
        volume = None
        area = None
        meta_path = os.path.join(pothole_dir, "pothole_meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    depth = meta.get("depth")
                    volume = meta.get("volume")
                    area = meta.get("area")
            except Exception:
                from exceptions.exceptions import StepPreconditionError
                raise StepPreconditionError(
                    "POTHOLE_META_JSON_READ_FAILED",
                    f"Failed to read {meta_path}",
                    context="build_road_json.parse_potholes",
                )

        # Position from GPS nearest to image timestamp (if image exists)
        lat = 0.0
        lng = 0.0
        if image_path is not None and segment_gps_data:
            ts = parse_float(os.path.splitext(os.path.basename(image_path))[0])
            if ts is not None:
                try:
                    gp = min(segment_gps_data, key=lambda d: abs(float(d.get("timestamp", 0.0)) - ts))
                    lat = float(gp.get("lat", 0.0))
                    lng = float(gp.get("lng", 0.0))
                except Exception:
                    pass

        potholes.append({
            "id": pothole_id,
            "road_id": road_id,
            "road_segment_id": road_segment_id,
            "lat": float(lat),
            "lng": float(lng),
            "depth": float(depth) if depth is not None else 0.0,
            "volume": float(volume) if volume is not None else 0.0,
            "area": float(area) if isinstance(area, (int, float)) else 0.0,
            "image": image_rel,
            "lidar_scan": lidar_rel
        })

    return potholes


# ------------------------------ Segment parsing ---------------------------- #


def detect_segment_survey_video(segment_dir: str) -> str:
    """Detect a survey video file from the segment folder (prefer immediate files).

    Priority:
    1) Immediate files in the segment_dir
    2) Known subfolders under the segment (excluding Potholes): Data/Survey video, Data/survey_video, Survey video

    Fallback: returns empty string if none found.
    """
    video_exts = (".mp4", ".mov", ".mkv", ".avi")

    # 1) Check immediate files in segment_dir
    immediate = find_immediate_files_with_extensions(segment_dir, video_exts)
    if immediate:
        logging.debug("Detected survey video (immediate) for %s: %s", segment_dir, immediate[0])
        return immediate[0]

    # 2) Check known subfolders (do not search whole segment recursively to avoid pothole files)
    subfolders = [
        os.path.join(segment_dir, "Data/Survey video"),
        os.path.join(segment_dir, "Data/survey_video"),
        os.path.join(segment_dir, "Survey video"),
    ]
    for base in subfolders:
        if os.path.isdir(base):
            vids = find_files_with_extensions(base, video_exts)
            if vids:
                logging.debug("Detected survey video (subfolder) for %s: %s", segment_dir, vids[0])
                return vids[0]

    logging.warning("No survey video found for segment: %s", segment_dir)
    return ""


def detect_segment_lidar_scan(segment_dir: str) -> str:
    """Detect segment-level lidar scan (.pcd), preferring files in the segment folder.

    Priority:
    1) Immediate .pcd files in segment_dir
    2) Known subfolders under the segment (excluding Potholes): Data/Lidar Scan, Data/lidar_scan, Lidar Scan

    Fallback: returns empty string if none found.
    """
    # 1) Check immediate .pcd in segment_dir
    immediate_pcds = find_immediate_files_with_extensions(segment_dir, (".pcd",))
    if immediate_pcds:
        logging.debug("Detected lidar scan (immediate) for %s: %s", segment_dir, immediate_pcds[0])
        return immediate_pcds[0]

    # 2) Check known subfolders
    subfolders = [
        os.path.join(segment_dir, "Data/Lidar Scan"),
        os.path.join(segment_dir, "Data/lidar_scan"),
        os.path.join(segment_dir, "Lidar Scan"),
    ]
    for base in subfolders:
        if os.path.isdir(base):
            pcds = find_files_with_extensions(base, (".pcd",))
            if pcds:
                logging.debug("Detected lidar scan (subfolder) for %s: %s", segment_dir, pcds[0])
                return pcds[0]

    logging.warning("No segment-level lidar .pcd found for segment: %s", segment_dir)
    return ""


def build_segment_payload(road_dir: str, segment_dir: str, road_id: str, target_roads_dir: str) -> Dict:
    """Build a single segment payload by reading assets from target tree."""
    logging.debug(
        "Building segment payload for directory: %s (name=%s)",
        segment_dir,
        os.path.basename(segment_dir),
    )
    road_segment_id = os.path.basename(segment_dir)
    # read start/end/length from segment_meta.json (written during copy phase)
    start_loc = None
    end_loc = None
    # survey/lidar from target tree
    survey_rel = ""
    lidar_rel = ""
    survey_guess = join_target(target_roads_dir, road_id, road_segment_id, "Data", "Survey video")
    lidar_guess = join_target(target_roads_dir, road_id, road_segment_id, "Data", "Lidar Scan")
    if os.path.isdir(survey_guess):
        vids = find_files_with_extensions(survey_guess, (".mp4", ".mov", ".mkv", ".avi"))
        if vids:
            survey_rel = to_relpath(vids[0], target_roads_dir)
    if os.path.isdir(lidar_guess):
        pcds = find_files_with_extensions(lidar_guess, (".pcd",))
        if pcds:
            lidar_rel = to_relpath(pcds[0], target_roads_dir)

    # Read normalized per-segment JSONs produced by copy phase
    gps_data = []
    seg_gps_json = os.path.join(segment_dir, "imu.json")
    if os.path.isfile(seg_gps_json):
        try:
            with open(seg_gps_json, "r") as f:
                gps_data = json.load(f)
        except Exception:
            from exceptions.exceptions import StepPreconditionError
            raise StepPreconditionError(
                "SEGMENT_GPS_JSON_READ_FAILED",
                f"Failed to read {seg_gps_json}",
                context="build_road_json.build_segment_payload",
            )

    depth_data = None
    depth_json = os.path.join(segment_dir, "segment_depth_timestamps.json")
    if os.path.isfile(depth_json):
        try:
            with open(depth_json, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    depth_data = data
        except Exception:
            from exceptions.exceptions import StepPreconditionError
            raise StepPreconditionError(
                "SEGMENT_DEPTH_JSON_READ_FAILED",
                f"Failed to read {depth_json}",
                context="build_road_json.build_segment_payload",
            )
    potholes = parse_potholes(segment_dir, road_segment_id, target_roads_dir, road_id, road_segment_id)

    length_in_km = 0.0
    seg_meta_json = os.path.join(segment_dir, "segment_meta.json")
    if os.path.isfile(seg_meta_json):
        try:
            with open(seg_meta_json, "r") as f:
                meta = json.load(f)
                length_in_km = float(meta.get("length_in_km", 0.0))
                if not start_loc:
                    start_loc = meta.get("start_loc") or start_loc
                if not end_loc:
                    end_loc = meta.get("end_loc") or end_loc
        except Exception:
            from exceptions.exceptions import StepPreconditionError
            raise StepPreconditionError(
                "SEGMENT_META_JSON_READ_FAILED",
                f"Failed to read {seg_meta_json}",
                context="build_road_json.build_segment_payload",
            )

    payload = {
        "id": road_segment_id,
        "road_id": road_id,
        "iri": 0.0,
        "location": "",
        "start_loc": start_loc or "",
        "end_loc": end_loc or "",
        "length_in_km": length_in_km,
        "survey_video": survey_rel,
        "lidar_scan": lidar_rel,
        "gps_data": gps_data,
        "depth_data": depth_data,
        "potholes": potholes,
    }
    return payload


# -------------------------------- Road parsing ----------------------------- #

def build_road_payload(road_dir: str, target_roads_dir: str) -> Dict:
    """Build a single road payload by assembling all its segments from target tree."""
    logging.debug(
        "Building road payload for directory: %s (name=%s)",
        road_dir,
        os.path.basename(road_dir),
    )
    road_id = os.path.basename(road_dir)
    segments: List[Dict] = []
    for segment_dir in list_immediate_subdirs(road_dir):
        if not os.path.isdir(os.path.join(segment_dir, "Potholes")):
            logging.debug("Skipping non-target segment dir: %s", segment_dir)
            continue
        logging.info("Detected segment: %s", segment_dir)
        segments.append(build_segment_payload(road_dir, segment_dir, road_id, target_roads_dir))

    return {
        "id": road_id,
        "name": "",
        "location": "",
        "road_segments": segments,
    }

def build_roads_json_from_target(target_roads_dir: str) -> Dict:
    """Build payload by reading the already-copied target tree (no copying)."""
    roads_payload: List[Dict] = []
    for road_dir in discover_target_roads(target_roads_dir):
        road_payload = build_road_payload(road_dir, target_roads_dir)
        roads_payload.append(road_payload)
    if not roads_payload:
        logging.warning("No road payloads produced. Check target tree: %s", target_roads_dir)
    return {"roads": roads_payload}


# ------------------------------------ main --------------------------------- #


from common.cli import add_config_arg, add_log_level_arg, parse_args_with_config, setup_logging

def build_parser(argv: Optional[List[str]] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build consolidated road JSON from target tree.")
    add_config_arg(parser); add_log_level_arg(parser)
    parser.add_argument("--target-roads-dir", help="Target Roads directory root; JSON paths are relative to here.")
    parser.add_argument("--output", help="Path to output JSON file.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args, cfg = parse_args_with_config(
        lambda: build_parser(argv),
        lambda c: dict(
            target_roads_dir=c.paths.target_roads_root,
            output=c.paths.json_out,
            log_level=c.logging.level,
        ),
        argv,
    )

    setup_logging(args.log_level)

    output_path = os.path.abspath(args.output)
    target_roads_dir = os.path.abspath(args.target_roads_dir)

    os.makedirs(target_roads_dir, exist_ok=True)
    logging.info("Target Roads directory: %s", target_roads_dir)

    payload = build_roads_json_from_target(target_roads_dir)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    logging.debug("Wrote JSON: %s", output_path)
    total_roads = len(payload.get("roads", []))
    total_segments = sum(len(r.get("road_segments", [])) for r in payload.get("roads", []))
    logging.info("Included roads: %d, segments: %d", total_roads, total_segments)

if __name__ == "__main__":
    main()


