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
# import DDD_pair_pothole application service
from assets.DDD_pair_pothole import ApplicationService

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def pair_pothole_assets(pcd_dir: Path, img_dir: Path, dest_dir: Path, start_id: int, zero_pad: int, move: bool, log_level: str) -> None:
    application_service = ApplicationService()
    application_service.run(img_dir, pcd_dir, dest_dir,
                            start_id, zero_pad, move, log_level)
