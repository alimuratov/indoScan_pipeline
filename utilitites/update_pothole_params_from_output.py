#!/usr/bin/env python3
"""update_pothole_params_from_output

Replace each pothole's depth and volume in a road JSON using values parsed from
its output.txt, and add an area field right after volume.

Sources in output.txt:
- depth: "Overall statistics" → "Mean depth: <value> m"
- volume: "Aggregated across clusters" → "Sum of volumes (convex hull approx.): <value> m³"
- area: "Aggregated across clusters" → "Sum of areas (convex hull): <value> m²"

Mapping pothole folders → JSON pothole objects is done by matching filename
stems (e.g., 1756369945.499594927) from a folder's .pcd/.jpg with the stem in
the JSON pothole's image/lidar_scan path.
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
    parser = argparse.ArgumentParser(description="Replace pothole depth/volume and add area from output.txt into road JSON")
    parser.add_argument("--json", required=True, help="Path to input JSON (e.g., all_segments.json)")
    parser.add_argument("--roads-root", required=True, help="Root containing road/segment/pothole folders")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <input>.updated.json)")
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
    """Index pothole objects by filename stem of their image or lidar_scan path."""
    index: Dict[str, dict] = {}
    for road in payload.get("roads", []):
        for seg in road.get("road_segments", []):
            for p in seg.get("potholes", []):
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
                    index[stem] = p
    return index


def is_pothole_dir(directory: Path) -> bool:
    try:
        names = {p.name.lower() for p in directory.iterdir() if p.is_file()}
    except Exception:
        return False
    has_pcd = any(n.endswith(".pcd") for n in names)
    has_img = any(any(n.endswith(ext) for ext in IMAGE_EXTS) for n in names)
    return has_pcd and has_img


def stem_from_pothole_dir(directory: Path) -> Optional[str]:
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


MEAN_DEPTH_RE = re.compile(r"Overall statistics[\s\S]*?Mean depth:\s*([0-9]*\.?[0-9]+)")
SUM_AREAS_RE = re.compile(r"Sum of areas\s*\(convex hull\):\s*([0-9]*\.?[0-9]+)")
SUM_VOLUMES_RE = re.compile(r"Sum of volumes\s*\(convex hull approx\.\):\s*([0-9]*\.?[0-9]+)")


def parse_output_file(output_path: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (mean_depth, sum_areas, sum_volumes) from output.txt or (None, None, None)."""
    try:
        text = output_path.read_text()
    except FileNotFoundError:
        return None, None, None

    md = None
    sa = None
    sv = None
    m = MEAN_DEPTH_RE.search(text)
    if m:
        try:
            md = float(m.group(1))
        except Exception:
            md = None
    m = SUM_AREAS_RE.search(text)
    if m:
        try:
            sa = float(m.group(1))
        except Exception:
            sa = None
    m = SUM_VOLUMES_RE.search(text)
    if m:
        try:
            sv = float(m.group(1))
        except Exception:
            sv = None
    return md, sa, sv


def updated_pothole_object(original: dict, depth: Optional[float], volume: Optional[float], area: Optional[float]) -> dict:
    """Return a new pothole dict with updated depth/volume and inserted area after volume.

    Preserves key order by rebuilding the dict.
    """
    new_obj: dict = {}
    for key, value in original.items():
        if key == "depth":
            new_obj[key] = float(depth) if depth is not None else value
            continue
        if key == "volume":
            new_obj[key] = float(volume) if volume is not None else value
            # Insert area immediately after volume
            if area is not None:
                new_obj["area"] = float(area)
            continue
        # Skip pre-existing area; it will be placed after volume
        if key == "area":
            continue
        new_obj[key] = value

    # If volume key didn't exist, add volume then area at the end in requested order
    if "volume" not in new_obj:
        if volume is not None:
            new_obj["volume"] = float(volume)
        if area is not None:
            new_obj["area"] = float(area)
    else:
        # If volume existed but area not inserted (area None), optionally keep old area
        if area is None and "area" in original and "area" not in new_obj:
            new_obj["area"] = original["area"]

    # If depth key didn't exist, append it
    if "depth" not in new_obj and depth is not None:
        new_obj["depth"] = float(depth)

    return new_obj


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

        mean_depth, sum_areas, sum_volumes = parse_output_file(pothole_dir / "output.txt")
        if mean_depth is None and sum_volumes is None and sum_areas is None:
            logging.debug("No parsable stats in: %s", pothole_dir / "output.txt")
            continue

        pothole_obj = index.get(stem)
        if pothole_obj is None:
            logging.warning("No JSON pothole matched for stem: %s (dir=%s)", stem, pothole_dir)
            misses += 1
            continue

        # Area fallback: if sum_areas missing but sum_volumes present and user phrasing implied volumes, use volumes
        area_value = sum_areas if sum_areas is not None else sum_volumes

        new_obj = updated_pothole_object(
            pothole_obj,
            depth=mean_depth,
            volume=sum_volumes,
            area=area_value,
        )

        pothole_obj.clear()
        pothole_obj.update(new_obj)
        updates += 1

    if updates == 0:
        logging.warning("No potholes updated. Check mapping and output.txt contents.")

    out_path: Path
    if args.in_place:
        out_path = json_path
    else:
        out_path = Path(args.out).resolve() if args.out else json_path.with_suffix(".updated.json")

    write_json(out_path, payload)
    logging.debug("Wrote updated JSON: %s (potholes updated: %d, misses: %d)", out_path, updates, misses)


if __name__ == "__main__":
    main()


