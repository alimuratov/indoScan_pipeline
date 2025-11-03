#!/usr/bin/env python3
"""pair_pothole_assets

Purpose:
- Group matched .pcd + image pairs (same filename stem) into numbered
  ``pothole_<id>`` folders under a destination directory.

- Standalone CLI utility

Inputs (CLI):
- --pcd-dir: Directory containing .pcd files.
- --images-dir: Directory containing image files (.jpg/.jpeg/.png/.bmp/.tif/.tiff).
- --dest: Destination directory where ``pothole_<id>`` folders will be created.
- --start-id: Starting integer for pothole IDs (default: 1).
- --zero-pad: Zero-pad width for IDs (default: 3 → ``pothole_001``).
- --move: Move files instead of copying (default: copy).
- --log-level: Logging level (DEBUG/INFO/WARNING/ERROR).

Outputs / Side effects:
- Creates ``pothole_<id>`` folders under ``--dest``.
- Copies or moves matched .pcd and image files into each folder.
- Logs warnings for unmatched items and an overall summary.
- Exits early with an error log if input directories are missing.

Notes / Assumptions:
- Matching is non-recursive and by exact filename stem equality.
- Only one image per .pcd (and vice versa) is handled per stem.
- Image extensions are limited by ``IMAGE_EXTS``; .pcd is required for pairing.

Usage:
    python scripts/assets/pair_pothole_assets.py \
        --pcd-dir /path/to/PCD \
        --images-dir /path/to/Images \
        --dest /path/to/output/potholes \
        --move
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Set

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def index_by_stem(directory: Path, exts: Set[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not directory.is_dir():
        return mapping
    for entry in directory.iterdir():
        if entry.is_file():
            ext = entry.suffix.lower()
            if exts and ext not in exts:
                continue
            mapping[entry.stem] = entry
    return mapping

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def pair_pothole_assets(pcd_dir: Path, img_dir: Path, dest_dir: Path, start_id: int, zero_pad: int, move: bool, log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    )

    ensure_dir(dest_dir)

    # check if dest_dir contains pothole_<id> folders and delete them
    for folder in dest_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("pothole_"):
            shutil.rmtree(folder)

    pcd_map = index_by_stem(pcd_dir, exts={".pcd"})
    img_map = index_by_stem(img_dir, exts=IMAGE_EXTS)

    pcd_names = set(pcd_map.keys())
    img_names = set(img_map.keys())
    # Sorting makes processing deterministic (i.e., the pairing order is consistent)
    matched = sorted(pcd_names & img_names)

    missing_img = sorted(pcd_names - img_names)
    missing_pcd = sorted(img_names - pcd_names)

    if missing_img:
        logging.error("Missing image for %d .pcd file(s): e.g. %s", len(missing_img), ", ".join(missing_img[:5]))
    if missing_pcd:
        logging.error("Missing .pcd for %d image file(s): e.g. %s", len(missing_pcd), ", ".join(missing_pcd[:5]))

    count = 0
    current_id = start_id

    for name in matched:
        pcd_path = pcd_map[name]
        img_path = img_map[name]

        folder = dest_dir / f"pothole_{current_id:0{zero_pad}d}"
        ensure_dir(folder)

        dest_pcd = folder / pcd_path.name
        dest_img = folder / img_path.name

        if move:
            shutil.move(str(pcd_path), dest_pcd)
            shutil.move(str(img_path), dest_img)
        else:
            shutil.copy2(str(pcd_path), dest_pcd)
            shutil.copy2(str(img_path), dest_img)

        logging.debug("%s → %s | %s → %s", pcd_path.name, dest_pcd, img_path.name, dest_img)
        count += 1
        current_id += 1

    logging.debug("Paired %d pothole folder(s) under: %s", count, dest_dir)
