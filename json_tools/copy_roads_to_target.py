#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any

from common.cli import add_config_arg, add_log_level_arg, parse_args_with_config, setup_logging
from common.parse import parse_pothole_parameters, ParseError  # type: ignore
from common.text import read_text_lines  # type: ignore
from common.ids import generate_prefixed_uuid  # type: ignore
from common.media import detect_segment_survey_video, detect_segment_lidar_scan, find_pothole_media
from common.fs import list_immediate_subdirs, copy_asset, ensure_parent_dir, CopyError
from common.discovery import discover_roads, is_segment_dir

def construct_segment_route_summary(segment_dir: str, segment_target: str, segment_gps_entries: List[Dict[str, Any]]) -> None:
    start_loc = ""
    end_loc = ""

    if segment_gps_entries:
        first = segment_gps_entries[0]
        last = segment_gps_entries[-1]
        start_loc = f"{first['lat']}, {first['lng']}"
        end_loc = f"{last['lat']}, {last['lng']}"

    length_km = 0.0
    length_json_src = os.path.join(segment_dir, "route_length.json")
    if os.path.isfile(length_json_src):
        try:
            with open(length_json_src, "r") as f:
                length_data = json.load(f)
                length_km = float(length_data.get("kilometers", 0.0))
        except Exception:
            pass
    segment_route_summary = {"start_loc": start_loc, "end_loc": end_loc, "length_in_km": length_km}
    segment_route_summary_path = os.path.join(segment_target, "segment_meta.json")
    with open(segment_route_summary_path, "w") as f:
        json.dump(segment_route_summary, f, indent=2)

def copy_segment_depth_timestamps(segment_dir: str, segment_target: str) -> None:
    depth_timestamps_src = os.path.join(segment_dir, "segment_depth_timestamps.json")
    if os.path.isfile(depth_timestamps_src):
        depth_timestamps_dst = os.path.join(segment_target, "segment_depth_timestamps.json")
        copy_asset(depth_timestamps_src, depth_timestamps_dst)


def copy_segment_imu(segment_dir: str, segment_target: str) -> None:
    imu_src = os.path.join(segment_dir, "imu.json")
    if os.path.isfile(imu_src):
        imu_dst = os.path.join(segment_target, "imu.json")
        copy_asset(imu_src, imu_dst)

def convert_gps_to_json_and_write(segment_dir: str, segment_target: str) -> List[Dict[str, Any]]:
    gps_json = []
    seg_gps_src = os.path.join(segment_dir, "gps.txt")
    if os.path.isfile(seg_gps_src):
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
                    gps_json.append({
                    "timestamp": ts,
                    "lat": lat,
                    "lng": lng,
                    "alt": alt,
                })
                except Exception:
                    pass
    gps_json.sort(key=lambda d: d["timestamp"]) if gps_json else None
    gps_json_path = os.path.join(segment_target, "segment_gps.json")
    ensure_parent_dir(gps_json_path)
    with open(gps_json_path, "w") as f:
        json.dump(gps_json, f, indent=2)

    return gps_json

def copy_segment_media(segment_dir: str, segment_target: str) -> None:
    survey_src = detect_segment_survey_video(segment_dir)
    if survey_src:
        survey_dst = os.path.join(segment_target, "Data", "Survey video", os.path.basename(survey_src))
        copy_asset(survey_src, survey_dst)
    lidar_src = detect_segment_lidar_scan(segment_dir)
    if lidar_src:
        lidar_dst = os.path.join(segment_target, "Data", "Lidar Scan", os.path.basename(lidar_src))
        copy_asset(lidar_src, lidar_dst)

def parse_and_dump_pothole_meta(pothole_dir: str, pothole_target: str) -> None:
    out_src = os.path.join(pothole_dir, "output.txt")
    if os.path.isfile(out_src):
        depth, volume, area = parse_pothole_parameters(out_src)
        
        meta = {
            "depth": float(depth) if isinstance(depth, (int, float)) else 0.0,
            "volume": float(volume) if isinstance(volume, (int, float)) else 0.0,
            "area": float(area) if isinstance(area, (int, float)) else 0.0,
        }

        meta_dst = os.path.join(pothole_target, "pothole_meta.json")
        ensure_parent_dir(meta_dst)
        with open(meta_dst, "w") as f:
            json.dump(meta, f, indent=2)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Manifest-driven copy: copy media and write normalized metas + manifest")
    add_config_arg(p); add_log_level_arg(p)
    p.add_argument("--source-roads-root", help="Source roads root (roads/<road>/<segment>/...)")
    p.add_argument("--target-roads-root", help="Target Roads directory root")
    p.add_argument("--manifest-out", help="Path to write copy manifest JSON")
    return p

def main() -> None:
    args, cfg = parse_args_with_config(
        build_parser,
        lambda cfg: dict(
            source_roads_root=cfg.paths.source_roads_root,
            target_roads_root=cfg.paths.target_roads_root,
            manifest_out=cfg.paths.manifest_out,
            log_level=cfg.logging.level,
        ),
    )

    setup_logging(args.log_level)

    source_roads_root = os.path.abspath(args.roads_root)
    target_roads_root = os.path.abspath(args.target_roads_dir)

    if not os.path.isdir(source_roads_root):
        from exceptions.exceptions import StepPreconditionError
        raise StepPreconditionError(
            "ROADS_ROOT_NOT_DIRECTORY",
            f"Roads root not found or not a directory: {source_roads_root}",
            context="copy_roads_to_target.main",
        )

    os.makedirs(target_roads_root, exist_ok=True)
    logging.info("Target Roads directory: %s", target_roads_root)

    manifest = {
        "roads_root": source_roads_root,
        "target_roads_dir": target_roads_root,
        "roads": [],
    }

    roads_count = 0
    segments_count = 0
    potholes_count = 0

    for source_road_dir in discover_roads(source_roads_root):
        road_id = generate_prefixed_uuid("rd")
        roads_count += 1
        road_manifest_entry = {"id": road_id, "source": source_road_dir, "segments": []}

        for source_segment_dir in list_immediate_subdirs(source_road_dir):
            try: 
                if not is_segment_dir(source_segment_dir):
                    continue

                segments_count += 1

                road_segment_id = generate_prefixed_uuid("rs")

                segment_entry = {
                    "id": road_segment_id,
                    "source": source_segment_dir,
                    "target": os.path.join(target_roads_root, road_id, road_segment_id),
                    "potholes": [],
                }

                # Copy segment-level media (Survey/Lidar if present at source)
                # We will compute meta (start/end/length) from GPS later
                os.makedirs(segment_entry["target"], exist_ok=True)

                copy_segment_media(source_segment_dir, segment_entry["target"])
                copy_segment_imu(source_segment_dir, segment_entry["target"])
                copy_segment_depth_timestamps(source_segment_dir, segment_entry["target"])

                pothole_dirs = [
                    candidate_path for candidate_path in list_immediate_subdirs(source_segment_dir)
                    if os.path.isdir(candidate_path) and os.path.basename(candidate_path).lower().startswith("pothole")
                ]

                for pothole_dir in pothole_dirs:
                    try:
                        potholes_count += 1
                        pothole_id = generate_prefixed_uuid("pt")
                        pothole_target_dir = os.path.join(segment_entry["target"], "Potholes", pothole_id)
                        pothole_image_src, pothole_pcd_src = find_pothole_media(pothole_dir)

                        # Copy media
                        if pothole_image_src:
                            pothole_image_dst = os.path.join(pothole_target_dir, "Image", os.path.basename(pothole_image_src))
                            copy_asset(pothole_image_src, pothole_image_dst)
                        if pothole_pcd_src:
                            pothole_pcd_dst = os.path.join(pothole_target_dir, "Lidar Scan", os.path.basename(pothole_pcd_src))
                            copy_asset(pothole_pcd_src, pothole_pcd_dst)

                        # Parse output.txt to normalized pothole_meta.json
                        parse_and_dump_pothole_meta(pothole_dir, pothole_target_dir)

                        # gps.txt resides at the segment level; handled after pothole loop

                        segment_entry["potholes"].append({
                            "id": pothole_id,
                            "source": pothole_dir,
                            "target": pothole_target_dir,
                            "image": os.path.basename(pothole_image_src) if pothole_image_src else "",
                            "pcd": os.path.basename(pothole_pcd_src) if pothole_pcd_src else "",
                        })

                    except ParseError as e:
                        logging.warning("ParseError in %s: %s", pothole_dir, e)
                        continue
                    except ValueError as e:
                        logging.warning("Media not found in %s: %s", pothole_dir, e)
                        continue
                    except CopyError as e:
                        logging.warning("CopyError in %s: %s", pothole_dir, e)
                        continue
                    except Exception as e:
                        from exceptions.exceptions import StepPreconditionError
                        raise StepPreconditionError(
                            "UNEXPECTED_POTHOLE_COPY_ERROR",
                            f"Unexpected error in {pothole_dir}: {e}",
                            context="copy_roads_to_target.main",
                        )
                
                road_manifest_entry["segments"].append(segment_entry)

                segment_gps_entries = convert_gps_to_json_and_write(source_segment_dir, segment_entry["target"])

                construct_segment_route_summary(source_segment_dir, segment_entry["target"], segment_gps_entries)

            except Exception as e:
                from exceptions.exceptions import StepPreconditionError
                raise StepPreconditionError(
                    "UNEXPECTED_SEGMENT_COPY_ERROR",
                    f"Unexpected error in {source_segment_dir}: {e}",
                    context="copy_roads_to_target.main",
                )

        manifest["roads"].append(road_manifest_entry)

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


