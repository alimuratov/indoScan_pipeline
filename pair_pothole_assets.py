#!/usr/bin/env python3
import argparse
import logging
import os
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pair .pcd and image files by name into pothole_<id> folders")
    parser.add_argument("--pcd-dir", required=True, help="Directory containing .pcd files")
    parser.add_argument("--images-dir", required=True, help="Directory containing image files")
    parser.add_argument("--dest", required=True, help="Destination directory to create pothole_<id> folders")
    parser.add_argument("--start-id", type=int, default=1, help="Starting ID for pothole folders (default: 1)")
    parser.add_argument("--zero-pad", type=int, default=3, help="Zero pad width for IDs (default: 3 → pothole_001)")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pcd_dir = Path(args.pcd_dir).resolve()
    img_dir = Path(args.images_dir).resolve()
    dest_dir = Path(args.dest).resolve()

    if not pcd_dir.is_dir():
        logging.error("PCD dir not found: %s", pcd_dir)
        return
    if not img_dir.is_dir():
        logging.error("Images dir not found: %s", img_dir)
        return

    ensure_dir(dest_dir)

    pcd_map = index_by_stem(pcd_dir, exts={".pcd"})
    img_map = index_by_stem(img_dir, exts=IMAGE_EXTS)

    pcd_names = set(pcd_map.keys())
    img_names = set(img_map.keys())
    matched = sorted(pcd_names & img_names)

    missing_img = sorted(pcd_names - img_names)
    missing_pcd = sorted(img_names - pcd_names)

    if missing_img:
        logging.warning("Missing image for %d .pcd file(s): e.g. %s", len(missing_img), ", ".join(missing_img[:5]))
    if missing_pcd:
        logging.warning("Missing .pcd for %d image file(s): e.g. %s", len(missing_pcd), ", ".join(missing_pcd[:5]))

    count = 0
    current_id = args.start_id
    for name in matched:
        pcd_path = pcd_map[name]
        img_path = img_map[name]

        folder = dest_dir / f"pothole_{current_id:0{args.zero_pad}d}"
        ensure_dir(folder)

        dest_pcd = folder / pcd_path.name
        dest_img = folder / img_path.name

        if args.move:
            shutil.move(str(pcd_path), dest_pcd)
            shutil.move(str(img_path), dest_img)
        else:
            shutil.copy2(str(pcd_path), dest_pcd)
            shutil.copy2(str(img_path), dest_img)

        logging.info("%s → %s | %s → %s", pcd_path.name, dest_pcd, img_path.name, dest_img)
        count += 1
        current_id += 1

    logging.info("Paired %d pothole folder(s) under: %s", count, dest_dir)


if __name__ == "__main__":
    main()


