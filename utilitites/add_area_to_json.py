#!/usr/bin/env python3
"""add_area_to_json

Augment pothole objects in a road JSON by adding an "area" field parsed from
each pothole folder's output.txt.

Mapping strategy:
- Match pothole folders to JSON pothole objects by filename stem shared between
  the folder's single .pcd/.jpg and the JSON's pothole image/lidar paths.
  (E.g., 1756369945.499594927.{jpg,pcd})

Usage:
  python scripts/add_area_to_json.py \
    --json /path/to/all_segments3.json \
    --roads-root /path/to/roads \
    --out /path/to/all_segments3.with_area.json

Use --in-place to overwrite the input JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add area from output.txt into pothole objects in a road JSON")
    parser.add_argument("--json", required=True, help="Path to input JSON (e.g., all_segments3.json)")
    parser.add_argument("--roads-root", required=True, help="Root containing road/segment/pothole folders")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <input>.with_area.json)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input JSON in place")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_pothole_index(payload: dict) -> Dict[str, dict]:
    """Index pothole objects by the filename stem in their image or lidar_scan.

    Returns a mapping: stem -> pothole_dict. If multiple potholes share a stem,
    later ones overwrite earlier; we log duplicates at DEBUG.
    """
    index: Dict[str, dict] = {}
    roads = payload.get("roads", [])
    for road in roads:
        segments = road.get("road_segments", [])
        for seg in segments:
            potholes = seg.get("potholes", [])
            for p in potholes:
                stem: Optional[str] = None
                for key in ("lidar_scan", "image"):
                    path = p.get(key)
                    if not path:
                        continue
                    name = os.path.basename(path)
                    stem = os.path.splitext(name)[0]
                    if stem:
                        break
                if stem:
                    if stem in index:
                        logging.debug("Duplicate stem in JSON index: %s", stem)
                    index[stem] = p
    return index


AREA_RE = re.compile(r"\bArea:\s*([0-9]*\.?[0-9]+)")


def extract_area_from_output(output_path: Path) -> Optional[float]:
    try:
        with open(output_path, "r") as f:
            for line in f:
                m = AREA_RE.search(line)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        return None
    except FileNotFoundError:
        return None
    return None


def is_pothole_dir(directory: Path) -> bool:
    try:
        names = {p.name.lower() for p in directory.iterdir() if p.is_file()}
    except Exception:
        return False
    has_pcd = any(n.endswith(".pcd") for n in names)
    has_img = any(any(n.endswith(ext) for ext in IMAGE_EXTS) for n in names)
    return has_pcd and has_img


def stem_from_pothole_dir(directory: Path) -> Optional[str]:
    # Prefer .pcd stem, else image stem
    pcds = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pcd"]
    if pcds:
        return pcds[0].stem
    imgs = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if imgs:
        return imgs[0].stem
    return None


def find_pothole_dirs(root: Path) -> List[Path]:
    if is_pothole_dir(root):
        return [root]
    out: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        d = Path(dirpath)
        if is_pothole_dir(d):
            out.append(d)
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    json_path = Path(args.json).resolve()
    roads_root = Path(args.roads_root).resolve()
    if not json_path.is_file():
        logging.error("JSON not found: %s", json_path)
        return
    if not roads_root.is_dir():
        logging.error("roads-root not found: %s", roads_root)
        return

    payload = load_json(json_path)
    index = build_pothole_index(payload)
    logging.info("Indexed %d potholes from JSON", len(index))

    pothole_dirs = find_pothole_dirs(roads_root)
    if not pothole_dirs:
        logging.warning("No pothole folders found under: %s", roads_root)
        return

    updates = 0
    misses = 0
    for pothole_dir in sorted(pothole_dirs):
        stem = stem_from_pothole_dir(pothole_dir)
        if not stem:
            logging.debug("No stem derived for: %s", pothole_dir)
            continue

        area = extract_area_from_output(pothole_dir / "output.txt")
        if area is None:
            logging.debug("No area found in: %s", pothole_dir / "output.txt")
            continue

        pothole_obj = index.get(stem)
        if pothole_obj is None:
            logging.warning("No JSON pothole matched for stem: %s (dir=%s)", stem, pothole_dir)
            misses += 1
            continue

        pothole_obj["area"] = float(area)
        updates += 1

    if updates == 0:
        logging.warning("No pothole areas updated. Check mapping and output.txt contents.")

    out_path: Path
    if args.in_place:
        out_path = json_path
    else:
        out_path = Path(args.out).resolve() if args.out else json_path.with_suffix(".with_area.json")

    write_json(out_path, payload)
    logging.info("updated JSON: %s (areas updated: %d, misses: %d)", out_path, updates, misses)


if __name__ == "__main__":
    main()


