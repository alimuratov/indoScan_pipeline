"""build_road_json

Generate a consolidated road-scanning JSON from a folder hierarchy of
roads → segments → potholes. The script also tries to integrate outputs from
the existing tools:
- estimate_depth.py → per-pothole output.txt (depth/volume lines)
- process_imu.py → imu.json entries (segment-level or collected from pothole dirs)
- retrieve_depth_timestamps.py → aggregated_depth_data.json (segment-level)

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
- depth_data: If aggregated_depth_data.json is present under the segment, we
  load it. Otherwise, we estimate it by parsing pothole image timestamps and
  depths from output.txt, computing a per-segment relative time using the
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


# ----------------------------- Utility helpers ----------------------------- #


def generate_prefixed_uuid(prefix: str) -> str:
    """Generate a random id string with a given prefix, e.g., "rd-<uuid>".

    Interaction: Used by road/segment/pothole builders to assign stable IDs
    referenced across the JSON structure.
    """
    return f"{prefix}-{uuid.uuid4()}"


def to_relpath(path: str, relative_to_dir: str) -> str:
    """Return path relative to a directory, preserving path separators.

    Interaction: Central helper to produce JSON paths relative to
    --target-roads-dir after assets are copied into the target tree.
    """
    try:
        return os.path.relpath(path, start=relative_to_dir)
    except Exception:
        logging.debug("to_relpath fallback for path=%s relative_to=%s", path, relative_to_dir)
        return path


def join_target(*parts: str) -> str:
    """Join path parts for target tree layout.

    Interaction: Used by segment/pothole builders to calculate destination
    asset paths (videos, lidar, images) under the target Roads directory.
    """
    return os.path.join(*parts)


def ensure_parent_dir(path: str) -> None:
    """Ensure parent directory exists for a given file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def copy_asset(src_path: str, dst_abs_path: str) -> bool:
    """Copy a file to destination, logging success/failure.

    Interaction: Called when copying survey videos, lidar scans, and images
    into the target tree so that JSON can reference them consistently.
    """
    try:
        ensure_parent_dir(dst_abs_path)
        shutil.copy2(src_path, dst_abs_path)
        logging.info("Copied asset: %s -> %s", src_path, dst_abs_path)
        return True
    except Exception:
        logging.exception("Failed to copy asset: %s -> %s", src_path, dst_abs_path)
        return False


def list_immediate_subdirs(parent_dir: str) -> List[str]:
    """List direct subdirectories of the given directory."""
    return [
        os.path.join(parent_dir, name)
        for name in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, name))
    ]


def find_files_with_extensions(root_dir: str, exts: Tuple[str, ...]) -> List[str]:
    """Recursively find files under root_dir with any of the extensions."""
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(exts):
                matches.append(os.path.join(dirpath, fname))
    return matches


def find_immediate_files_with_extensions(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    """Find files in dir_path (non-recursive) that end with any given extensions."""
    matches: List[str] = []
    try:
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(exts):
                matches.append(fpath)
    except Exception:
        logging.exception("Failed listing directory for immediate files: %s", dir_path)
    return matches


def find_first_matching_file(root_dir: str, patterns: Tuple[str, ...]) -> Optional[str]:
    """Find first file whose lowercase name contains any pattern.

    Note: Not currently used by the main flow but kept for potential extension.
    """
    patterns_lower = tuple(p.lower() for p in patterns)
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            low = fname.lower()
            if any(p in low for p in patterns_lower):
                return os.path.join(dirpath, fname)
    return None


def read_text_lines(path: str) -> List[str]:
    """Read all lines from a text file.

    Fallback logs: if file cannot be opened, the caller handles the exception
    and logs details (e.g., gps.txt or output.txt not found).
    """
    with open(path, "r") as f:
        return f.readlines()


def parse_float(s: str) -> Optional[float]:
    """Parse string to float, returning None on failure."""
    try:
        return float(s)
    except Exception:
        return None


# ------------------------------- GPS parsing ------------------------------- #


@dataclass
class GpsPoint:
    timestamp: float
    lat: float
    lng: float
    alt: Optional[float]


def parse_gps_file(gps_path: str) -> List[GpsPoint]:
    """Parse gps.txt with format: "#timestamp latitude longitude altitude" per line."""
    points: List[GpsPoint] = []
    try:
        for line in read_text_lines(gps_path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            ts = parse_float(parts[0])
            lat = parse_float(parts[1])
            lng = parse_float(parts[2])
            alt = parse_float(parts[3]) if len(parts) >= 4 else None
            if ts is None or lat is None or lng is None:
                continue
            points.append(GpsPoint(timestamp=ts, lat=lat, lng=lng, alt=alt))
    except FileNotFoundError:
        logging.debug("gps.txt not found: %s", gps_path)
    return points


def pick_nearest_gps(points: List[GpsPoint], target_ts: float) -> Optional[GpsPoint]:
    """Pick GPS point whose timestamp is closest to the target timestamp."""
    if not points:
        return None
    return min(points, key=lambda p: abs(p.timestamp - target_ts))


def collect_segment_gps(segment_dir: str) -> List[GpsPoint]:
    """Aggregate GPS points from all gps.txt files within the segment.

    Interaction: Used by segment coordinate computation and pothole geotagging.
    """
    logging.debug("Collecting GPS points for segment: %s", segment_dir)
    all_points: List[GpsPoint] = []
    for dirpath, _, filenames in os.walk(segment_dir):
        for fname in filenames:
            if fname.lower() == "gps.txt":
                path = os.path.join(dirpath, fname)
                logging.debug("Parsing GPS file: %s", path)
                all_points.extend(parse_gps_file(path))
    return sorted(all_points, key=lambda p: p.timestamp)


def compute_segment_start_end_loc(segment_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Compute segment start/end lat,lng from aggregated GPS points.

    Fallback: Logs a warning and returns (None, None) if no GPS is found.
    """
    points = collect_segment_gps(segment_dir)
    if not points:
        logging.warning("No GPS points found for segment: %s", segment_dir)
        return None, None
    start = points[0]
    end = points[-1]
    logging.debug("Segment %s start/end loc computed: (%s, %s) -> (%s, %s)", segment_dir, start.lat, start.lng, end.lat, end.lng)
    return f"{start.lat}, {start.lng}", f"{end.lat}, {end.lng}"


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
            logging.exception("Failed to read imu.json: %s", path)
            continue
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


def load_or_build_depth_data(segment_dir: str) -> List[Dict]:
    """Load aggregated depth data if present, else build from pothole folders.

    Building heuristic:
    - For each pothole, pick one image file and derive its timestamp from the
      filename (strip extension). Compute a segment-wide t0 as the earliest such
      timestamp across potholes.
    - Parse depth from that pothole's output.txt and compute relative time as
      ts - t0.
    """
    # Try load aggregated_depth_data.json first
    for dirpath, _, filenames in os.walk(segment_dir):
        for fname in filenames:
            if fname.lower() == "aggregated_depth_data.json":
                try:
                    with open(os.path.join(dirpath, fname), "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # Normalize
                            norm: List[Dict] = []
                            for item in data:
                                if not isinstance(item, dict):
                                    continue
                                depth = item.get("pothole_depth")
                                ts = str(item.get("video_timestamp", ""))
                                if isinstance(depth, (int, float)):
                                    norm.append({
                                        "pothole_depth": float(depth),
                                        "video_timestamp": ts,
                                    })
                            norm.sort(key=lambda d: float(d["video_timestamp"]))
                            return norm
                except Exception:
                    logging.exception("Failed to load aggregated_depth_data.json in %s", dirpath)

    return None


# --------------------------- Pothole parsing (segment) --------------------- #


def parse_potholes(segment_dir: str, road_segment_id: str, target_roads_dir: str, road_id: str, segment_id: str) -> List[Dict]:
    """Build pothole array for a segment: copy assets and extract metadata.

    Interaction: Called by segment builder to populate the "potholes" array.
    It copies image/pcd assets to the target tree and maps GPS to nearest
    image timestamp when available.
    """
    # filter out folders not starting with "pothole"
    pothole_dirs = [
        p for p in list_immediate_subdirs(segment_dir)
        if os.path.isdir(p) and os.path.basename(p).lower().startswith("pothole")
    ]
    logging.debug("Pothole directories under %s: %s", segment_dir, pothole_dirs)
    all_segment_gps = collect_segment_gps(segment_dir)
    potholes: List[Dict] = []

    for pothole_dir in pothole_dirs:
        pothole_id = generate_prefixed_uuid("pt")

        # Files inside pothole
        images = [
            os.path.join(pothole_dir, f)
            for f in os.listdir(pothole_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]
        image_path = images[0] if images else None
        image_rel = ""
        if image_path:
            image_dest_abs = join_target(
                target_roads_dir, road_id, segment_id, "Potholes", pothole_id, "Image", os.path.basename(image_path)
            )
            copy_asset(image_path, image_dest_abs)
            image_rel = to_relpath(image_dest_abs, target_roads_dir)
        else:
            logging.warning("No image found in pothole: %s", pothole_dir)

        pcds = [
            os.path.join(pothole_dir, f)
            for f in os.listdir(pothole_dir)
            if f.lower().endswith(".pcd")
        ]
        lidar_rel = ""
        if pcds:
            lidar_dest_abs = join_target(
                target_roads_dir, road_id, segment_id, "Potholes", pothole_id, "Lidar Scan", os.path.basename(pcds[0])
            )
            copy_asset(pcds[0], lidar_dest_abs)
            lidar_rel = to_relpath(lidar_dest_abs, target_roads_dir)
        else:
            logging.warning("No .pcd found in pothole: %s", pothole_dir)

        # Depth/volume from output.txt
        depth, volume = parse_depth_and_volume_from_output(os.path.join(pothole_dir, "output.txt"))

        # Position from GPS nearest to image timestamp (if image exists)
        lat = 0.0
        lng = 0.0
        if image_path is not None:
            ts = parse_float(os.path.splitext(os.path.basename(image_path))[0])
            if ts is not None:
                gp = pick_nearest_gps(all_segment_gps, ts)
                if gp is not None:
                    lat = gp.lat
                    lng = gp.lng

        potholes.append({
            "id": pothole_id,
            "road_id": road_id,
            "road_segment_id": road_segment_id,
            "lat": float(lat),
            "lng": float(lng),
            "depth": float(depth) if depth is not None else 0.0,
            "volume": float(volume) if volume is not None else 0.0,
            "image": image_rel,
            "lidar_scan": lidar_rel,
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


def is_pothole_dir(path: str) -> bool:
    """Heuristic: dir containing an image, a .pcd, or an output.txt.

    Interaction: Used by segment detection to decide if a subdirectory implies
    this is a segment containing pothole subfolders.
    """
    try:
        names = set(os.listdir(path))
    except Exception:
        return False
    has_artifact = any(
        any(name.lower().endswith(ext) for ext in (".jpg", ".png", ".pcd"))
        for name in names
    ) or ("output.txt" in names)
    return has_artifact


def is_segment_dir(path: str) -> bool:
    """Return True if directory has at least one pothole-like subdir."""
    try:
        for child in list_immediate_subdirs(path):
            if is_pothole_dir(child):
                return True
    except Exception:
        return False
    return False


def build_segment_payload(road_dir: str, segment_dir: str, road_id: str, target_roads_dir: str) -> Dict:
    """Build a single segment payload with copied assets and extracted data.

    Interaction: Called by road builder, which passes the generated road ID and
    target directory. This function computes segment start/end, copies segment
    assets, aggregates imu/depth data, and constructs the potholes list.
    """
    logging.debug(
        "Building segment payload for directory: %s (name=%s)",
        segment_dir,
        os.path.basename(segment_dir),
    )
    road_segment_id = generate_prefixed_uuid("rs")
    logging.debug(
        "Assigned road_segment_id=%s for segment directory: %s",
        road_segment_id,
        os.path.basename(segment_dir),
    )
    start_loc, end_loc = compute_segment_start_end_loc(segment_dir)
    survey_video = detect_segment_survey_video(segment_dir)
    lidar_scan = detect_segment_lidar_scan(segment_dir)
    gps_data = load_imu_entries_from_segment(segment_dir)
    depth_data = load_or_build_depth_data(segment_dir)
    potholes = parse_potholes(segment_dir, road_segment_id, target_roads_dir, road_id, road_segment_id)
    length_json = find_first_matching_file(segment_dir, ("route_length.json",))
    length_in_km = 0.0
    if length_json:
        with open(length_json, "r") as f:
            data = json.load(f)
            length_in_km = data.get("kilometers", 0.0)

    # Copy segment-level assets into target structure (original destinations):
    # roads/<road_id>/<segment_id>/Data/Survey video/<file>
    # roads/<road_id>/<segment_id>/Data/Lidar Scan/<file>
    survey_rel = ""
    if survey_video:
        survey_dest_abs = join_target(
            target_roads_dir, road_id, road_segment_id, "Data", "Survey video", os.path.basename(survey_video)
        )
        copy_asset(survey_video, survey_dest_abs)
        survey_rel = to_relpath(survey_dest_abs, target_roads_dir)

    lidar_rel = ""
    if lidar_scan:
        lidar_dest_abs = join_target(
            target_roads_dir, road_id, road_segment_id, "Data", "Lidar Scan", os.path.basename(lidar_scan)
        )
        copy_asset(lidar_scan, lidar_dest_abs)
        lidar_rel = to_relpath(lidar_dest_abs, target_roads_dir)

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


def is_road_dir(path: str) -> bool:
    """Return True if directory has at least one segment-like subdir."""
    try:
        for child in list_immediate_subdirs(path):
            if is_segment_dir(child):
                return True
    except Exception:
        return False
    return False


def build_road_payload(road_dir: str, target_roads_dir: str) -> Dict:
    """Build a single road payload by assembling all its segments."""
    logging.debug(
        "Building road payload for directory: %s (name=%s)",
        road_dir,
        os.path.basename(road_dir),
    )
    road_id = generate_prefixed_uuid("rd")
    logging.debug(
        "Assigned road_id=%s for road directory: %s",
        road_id,
        os.path.basename(road_dir),
    )
    segments: List[Dict] = []
    for segment_dir in list_immediate_subdirs(road_dir):
        if not is_segment_dir(segment_dir):
            logging.debug("Skipping non-segment dir: %s", segment_dir)
            continue
        logging.info("Detected segment: %s", segment_dir)
        segments.append(build_segment_payload(road_dir, segment_dir, road_id, target_roads_dir))

    return {
        "id": road_id,
        "name": "",
        "location": "",
        "road_segments": segments,
    }


def discover_roads(roads_root: str) -> List[str]:
    """Discover road directories under the provided root.

    Fallbacks:
    - Logs an info when no roads are found so the user can adjust the input.
    """
    roads: List[str] = []
    for maybe_road in list_immediate_subdirs(roads_root):
        if is_road_dir(maybe_road):
            logging.info("Detected road: %s", maybe_road)
            roads.append(maybe_road)
        else:
            logging.debug("Skipping non-road dir: %s", maybe_road)

    if not roads:
        logging.warning("No roads detected under: %s", roads_root)
    return roads


def build_roads_json(roads_root: str, output_path: str, target_roads_dir: str) -> Dict:
    """Top-level builder: discovers roads, builds payload, returns JSON dict.

    Interaction: Orchestrates the entire pipeline; road → segments → potholes,
    including copying assets and computing derived fields.
    """
    # All asset paths in JSON are relative to target_roads_dir
    roads_payload: List[Dict] = []
    for road_dir in discover_roads(roads_root):
        roads_payload.append(build_road_payload(road_dir, target_roads_dir))
    if not roads_payload:
        logging.warning("No road payloads produced. Check --roads-root path and structure.")
    return {"roads": roads_payload}


# ------------------------------------ main --------------------------------- #


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consolidated road JSON from folder structure.")
    parser.add_argument("--roads-root", required=True, help="Root directory containing road folders.")
    parser.add_argument("--output", default="road-scanning-POC-json.json", help="Path to output JSON file.")
    parser.add_argument("--target-roads-dir", default=os.path.join("data", "Roads"), help="Target Roads directory root; JSON paths are relative to here.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    roads_root = os.path.abspath(args.roads_root)
    output_path = os.path.abspath(args.output)
    target_roads_dir = os.path.abspath(args.target_roads_dir)

    if not os.path.isdir(roads_root):
        logging.error("Roads root not found or not a directory: %s", roads_root)
        sys.exit(1)

    os.makedirs(target_roads_dir, exist_ok=True)
    logging.info("Target Roads directory: %s", target_roads_dir)

    logging.info("Discovering roads under: %s", roads_root)
    payload = build_roads_json(roads_root, output_path, target_roads_dir)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    logging.info("Wrote JSON: %s", output_path)
    total_roads = len(payload.get("roads", []))
    total_segments = sum(len(r.get("road_segments", [])) for r in payload.get("roads", []))
    logging.info("Included roads: %d, segments: %d", total_roads, total_segments)


if __name__ == "__main__":
    main()


