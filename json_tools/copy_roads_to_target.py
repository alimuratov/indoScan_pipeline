#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Manifest-driven copy: copy media and write normalized metas + manifest")
    p.add_argument("--roads-root", required=True, help="Source roads root (roads/<road>/<segment>/...)")
    p.add_argument("--target-roads-dir", default=os.path.join("data", "Roads"), help="Target Roads directory root")
    p.add_argument("--manifest-out", default=os.path.join("data", "copy_manifest.json"), help="Path to write copy manifest JSON")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    p.add_argument("--config", help="Path to config file")
    return p


def main() -> None:
    """
    Parsing arguments in two passes:
    1. First pass: set config parameters as defaults
    2. Second pass: parse arguments from the command line (overriding defaults if provided)
    """
    p = build_parser()
    cfg_path = p.parse_known_args()[0].config
    
    from common.config import load_config
    cfg = load_config(cfg_path)

    p.set_defaults(
        roads_root=cfg.paths.roads_root,
        target_roads_dir=cfg.paths.target_roads_dir,
        manifest_out=cfg.paths.manifest_out,
        log_level=cfg.logging.level,
    )

    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="% (asctime)s % (levelname)s % (message)s".replace(" ", ""),
    )

    script_dir = Path(__file__).resolve().parent
    scripts_root = script_dir.parent
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))

    from common.parse import parse_depth_and_volume_from_output  # type: ignore
    from common.text import read_text_lines  # type: ignore
    from common.ids import generate_prefixed_uuid  # type: ignore
    from common.media import detect_segment_survey_video, detect_segment_lidar_scan  # type: ignore
    from common.fs import list_immediate_subdirs, copy_asset, ensure_parent_dir
    from common.discovery import discover_roads, is_segment_dir

    roads_root = os.path.abspath(args.roads_root)
    target_roads_dir = os.path.abspath(args.target_roads_dir)

    if not os.path.isdir(roads_root):
        logging.error("Roads root not found or not a directory: %s", roads_root)
        sys.exit(1)

    os.makedirs(target_roads_dir, exist_ok=True)
    logging.info("Target Roads directory: %s", target_roads_dir)

    manifest = {
        "roads_root": roads_root,
        "target_roads_dir": target_roads_dir,
        "roads": [],
    }

    roads_count = 0
    segments_count = 0
    potholes_count = 0

    for road_dir in discover_roads(roads_root):
        road_id = generate_prefixed_uuid("rd")
        roads_count += 1
        road_entry = {"id": road_id, "source": road_dir, "segments": []}

        for segment_dir in list_immediate_subdirs(road_dir):
            if not is_segment_dir(segment_dir):
                continue
            segments_count += 1
            road_segment_id = generate_prefixed_uuid("rs")
            seg_entry = {
                "id": road_segment_id,
                "source": segment_dir,
                "target": os.path.join(target_roads_dir, road_id, road_segment_id),
                "potholes": [],
            }

            # Copy segment-level media (Survey/Lidar if present at source)
            # We will compute meta (start/end/length) from GPS later
            os.makedirs(seg_entry["target"], exist_ok=True)
            survey_src = detect_segment_survey_video(segment_dir)
            if survey_src:
                survey_dst = os.path.join(seg_entry["target"], "Data", "Survey video", os.path.basename(survey_src))
                copy_asset(survey_src, survey_dst)
            lidar_src = detect_segment_lidar_scan(segment_dir)
            if lidar_src:
                lidar_dst = os.path.join(seg_entry["target"], "Data", "Lidar Scan", os.path.basename(lidar_src))
                copy_asset(lidar_src, lidar_dst)

            # Copy imu.json if present
            imu_src = os.path.join(segment_dir, "imu.json")
            if os.path.isfile(imu_src):
                imu_dst = os.path.join(seg_entry["target"], "imu.json")
                copy_asset(imu_src, imu_dst)

            # Copy depth timestamps JSON if present; normalize name in target
            src_depth_json = None
            for candidate in ("segment_depth_timestamps.json", "aggregated_depth_data.json"):
                cand_path = os.path.join(segment_dir, candidate)
                if os.path.isfile(cand_path):
                    src_depth_json = cand_path
                    break
            if src_depth_json:
                depth_dst = os.path.join(seg_entry["target"], "segment_depth_timestamps.json")
                copy_asset(src_depth_json, depth_dst)

            # Prepare per-segment GPS; will read from segment-level gps.txt
            merged_gps = []  # list of {timestamp, lat, lng, alt}

            pothole_dirs = [
                p for p in list_immediate_subdirs(segment_dir)
                if os.path.isdir(p) and os.path.basename(p).lower().startswith("pothole")
            ]
            for pothole_dir in pothole_dirs:
                potholes_count += 1
                pothole_id = generate_prefixed_uuid("pt")
                pothole_target = os.path.join(seg_entry["target"], "Potholes", pothole_id)
                img_src = None
                pcd_src = None
                for fname in os.listdir(pothole_dir):
                    low = fname.lower()
                    fpath = os.path.join(pothole_dir, fname)
                    if os.path.isfile(fpath) and low.endswith((".jpg", ".png")) and img_src is None:
                        img_src = fpath
                    elif os.path.isfile(fpath) and low.endswith(".pcd") and pcd_src is None:
                        pcd_src = fpath

                # Copy media
                if img_src:
                    img_dst = os.path.join(pothole_target, "Image", os.path.basename(img_src))
                    copy_asset(img_src, img_dst)
                if pcd_src:
                    pcd_dst = os.path.join(pothole_target, "Lidar Scan", os.path.basename(pcd_src))
                    copy_asset(pcd_src, pcd_dst)

                # Parse output.txt to normalized pothole_meta.json
                depth = None
                volume = None
                area = None
                out_src = os.path.join(pothole_dir, "output.txt")
                if os.path.isfile(out_src):
                    d, v = parse_depth_and_volume_from_output(out_src)
                    depth = d if d is not None else None
                    volume = v if v is not None else None
                    # Try parse area if present
                    try:
                        for line in read_text_lines(out_src):
                            if "Sum of areas" in line or "Area:" in line:
                                import re as _re
                                nums = _re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
                                if nums:
                                    area = float(nums[0])
                                    break
                    except Exception:
                        pass
                    meta = {
                        "depth": float(depth) if isinstance(depth, (int, float)) else 0.0,
                        "volume": float(volume) if isinstance(volume, (int, float)) else 0.0,
                        "area": float(area) if isinstance(area, (int, float)) else 0.0,
                    }
                    meta_dst = os.path.join(pothole_target, "pothole_meta.json")
                    ensure_parent_dir(meta_dst)
                    with open(meta_dst, "w") as f:
                        json.dump(meta, f, indent=2)

                # gps.txt resides at the segment level; handled after pothole loop

                seg_entry["potholes"].append({
                    "id": pothole_id,
                    "source": pothole_dir,
                    "target": pothole_target,
                    "image": os.path.basename(img_src) if img_src else "",
                    "pcd": os.path.basename(pcd_src) if pcd_src else "",
                })

            # Populate per-segment GPS from segment-level gps.txt and write metas
            seg_gps_src = os.path.join(segment_dir, "gps.txt")
            if os.path.isfile(seg_gps_src):
                try:
                    for line in read_text_lines(seg_gps_src):
                        s = line.strip()
                        if not s or s.startswith("#"):
                            continue
                        parts = s.split()
                        if len(parts) >= 3:
                            try:
                                ts = float(parts[0])
                                lat = float(parts[1])
                                lng = float(parts[2])
                                alt = float(parts[3]) if len(parts) >= 4 else None
                                merged_gps.append({
                                    "timestamp": ts,
                                    "lat": lat,
                                    "lng": lng,
                                    "alt": alt,
                                })
                            except Exception:
                                pass
                except Exception:
                    logging.exception("Failed reading segment gps: %s", seg_gps_src)
            merged_gps.sort(key=lambda d: d["timestamp"]) if merged_gps else None
            seg_gps_json = os.path.join(seg_entry["target"], "segment_gps.json")
            ensure_parent_dir(seg_gps_json)
            with open(seg_gps_json, "w") as f:
                json.dump(merged_gps, f, indent=2)

            # segment_meta.json: start_loc, end_loc, length_in_km (optional)
            start_loc = ""
            end_loc = ""
            if merged_gps:
                first = merged_gps[0]
                last = merged_gps[-1]
                start_loc = f"{first['lat']}, {first['lng']}"
                end_loc = f"{last['lat']}, {last['lng']}"
            length_km = 0.0
            length_json_src = os.path.join(segment_dir, "route_length.json")
            if os.path.isfile(length_json_src):
                try:
                    with open(length_json_src, "r") as f:
                        d = json.load(f)
                        length_km = float(d.get("kilometers", 0.0))
                except Exception:
                    pass
            seg_meta = {"start_loc": start_loc, "end_loc": end_loc, "length_in_km": length_km}
            seg_meta_json = os.path.join(seg_entry["target"], "segment_meta.json")
            with open(seg_meta_json, "w") as f:
                json.dump(seg_meta, f, indent=2)

            road_entry["segments"].append(seg_entry)

        manifest["roads"].append(road_entry)

    # Write manifest
    manifest_out = os.path.abspath(args.manifest_out)
    ensure_parent_dir(manifest_out)
    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)

    logging.info(
        "Copy complete. Roads: %d, segments: %d, potholes: %d. Manifest: %s",
        roads_count, segments_count, potholes_count, manifest_out
    )


if __name__ == "__main__":
    main()


