"""CLI entrypoint for assets_pairing.

Pairs timestamped images with timestamped point clouds (PCDs) by filename stem,
and writes them into `pothole_###/` folders under an output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from assets_pairing.application.use_case import PairAssetsUseCase
from assets_pairing.infrastructure.filesystem import (
    FilesystemAssetSource,
    FilesystemSnapshotStore,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pair images and PCDs by filename stem into pothole_* folders."
    )
    p.add_argument("--image-dir", type=Path, required=True,
                   help="Directory containing timestamped images (e.g. 123.45.jpg).")
    p.add_argument("--pcd-dir", type=Path, required=True,
                   help="Directory containing timestamped .pcd files (e.g. 123.45.pcd).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Destination segment directory where pothole_* folders are created.")
    p.add_argument("--start-id", type=int, default=1,
                   help="Starting pothole id (default: 1).")
    p.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default: copy).",
    )
    p.add_argument(
        "--zero-pad",
        type=int,
        default=3,
        help="Zero pad width for pothole ids (default: 3 -> pothole_001).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    image_dir: Path = args.image_dir.resolve()
    pcd_dir: Path = args.pcd_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not image_dir.is_dir():
        raise SystemExit(f"--image-dir does not exist: {image_dir}")
    if not pcd_dir.is_dir():
        raise SystemExit(f"--pcd-dir does not exist: {pcd_dir}")

    asset_source = FilesystemAssetSource(image_dir=image_dir, pcd_dir=pcd_dir)
    snapshot_store = FilesystemSnapshotStore(
        destination_directory_path=output_dir,
        zero_pad=int(args.zero_pad),
        move=bool(args.move),
    )

    summary = PairAssetsUseCase(
        asset_source=asset_source, snapshot_store=snapshot_store
    ).run(start_id=int(args.start_id))

    print("Pairing complete.")
    print(f"  written_pairs={summary.written_pairs}")
    print(f"  matched={len(summary.pairing.matched_keys)}")
    print(f"  missing_images={len(summary.pairing.missing_images)}")
    print(f"  missing_pcds={len(summary.pairing.missing_pcds)}")
    if summary.pairing.missing_images:
        print("  missing_images_stems=", summary.pairing.missing_images)
    if summary.pairing.missing_pcds:
        print("  missing_pcds_stems=", summary.pairing.missing_pcds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
