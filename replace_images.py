#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def main():
    parser = argparse.ArgumentParser(description="Replace raw images with annotated ones by matching filenames.")
    parser.add_argument("raw_dir", help="Folder with raw images to be replaced")
    parser.add_argument("annotated_dir", help="Folder with annotated images (same filenames)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    ann_dir = Path(args.annotated_dir)

    if not raw_dir.is_dir() or not ann_dir.is_dir():
        raise SystemExit("Both arguments must be existing directories.")

    replaced = 0
    missing = 0

    for raw_file in raw_dir.iterdir():
        if not raw_file.is_file():
            continue
        if raw_file.suffix.lower() not in IMAGE_EXTS:
            continue
        ann_file = ann_dir / raw_file.name
        if ann_file.is_file():
            shutil.copy2(ann_file, raw_file)
            replaced += 1
        else:
            missing += 1

    print(f"Replaced: {replaced} image(s). Missing annotated counterpart: {missing}.")
    
if __name__ == "__main__":
    main()