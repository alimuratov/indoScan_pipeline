#!/usr/bin/env python3
"""replace_images

Purpose:
- Replace each image file in a "raw" images directory with the annotated
  counterpart from another directory by matching filenames.

- Standalone CLI utility

Inputs (CLI):
- raw_dir: Path to folder containing original/raw images.
- annotated_dir: Path to folder containing annotated images with the same filenames.

Outputs/Side effects:
- Overwrites files in raw_dir with files from annotated_dir when a matching filename is found.
- Preserves file metadata where supported (via shutil.copy2).
- Prints a concise summary to stdout: "Replaced: <N>. Missing annotated counterpart: <M>."
- Exits with a non-zero status if provided directories do not exist.

Notes/Assumptions:
- Only files with extensions in IMAGE_EXTS are considered; others are ignored.
- Directory traversal is non-recursive (only top-level files in each directory).

Usage:
    python scripts/media/replace_images.py /path/to/raw_images /path/to/annotated_images
"""
import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def main():
    """Replace raw images with annotated counterparts by matching filenames.

    Parses CLI arguments, validates input directories, copies annotated files
    over raw files when names match, and prints a summary. Designed for
    one-off CLI use; returns None and raises SystemExit on invalid inputs.
    """
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