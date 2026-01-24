import os
import shutil
from pathlib import Path

import pytest

# run this for debugging
"RUN_REAL_TESTS=1 pytest -q tests/test_real_assets_pairing_smoke.py -s"

# RUN_REALT_TESTS=1 means that we're intentionally running side-effectful tests that mutate real_test fixtures.
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_TESTS") != "1",
    reason="Set RUN_REAL_TESTS=1 to run tests that mutate real_test fixtures.",
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_ROOT = REPO_ROOT / "scripts" / "src" / "real_test"
SOURCE_ROOT2 = REAL_TEST_ROOT / "source_roads_root2"
SEG_002 = SOURCE_ROOT2 / "seg-002"


def _reset_source_roads_root(*, source_root: Path, pothole_prefix: str = "pothole_") -> None:
    """Reset a source_roads_root-like directory to a pre-pairing state.

    These smoke tests are intentionally side-effectful when RUN_REAL_TESTS=1,
    so we keep the reset logic local (and avoid relying on a separate script).
    """
    source_root = source_root.resolve()

    segment_generated_files = {
        "imu.json",
        "route_length.json",
        "segment_depth_timestamps.json",
        "segment_gps.json",
        "segment_meta.json",
        "output_video.mp4",
    }

    if source_root.is_dir():
        for segment_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
            # Delete pothole_* folders (including exported *_c### folders)
            for child in segment_dir.iterdir():
                if child.is_dir() and child.name.startswith(pothole_prefix):
                    shutil.rmtree(child)

            # Delete known generated files
            for name in segment_generated_files:
                try:
                    (segment_dir / name).unlink()
                except FileNotFoundError:
                    pass

    # Remove any nested data2 export roots created under this source root.
    # Identified by the presence of copy_manifest.json + Roads/.
    if source_root.is_dir():
        for manifest in source_root.rglob("copy_manifest.json"):
            export_root = manifest.parent
            if (export_root / "Roads").is_dir():
                shutil.rmtree(export_root)


def test_real_assets_pairing_creates_pothole_folders() -> None:
    """Pair legacy Images/ + PCDs/ into pothole_* folders under the segment dir.

    This matches the intended pipeline: pairing is side-effectful and creates pothole_* folders.
    """
    if not SEG_002.is_dir():
        pytest.skip(f"Missing fixture: {SEG_002}")

    # Reset to baseline (idempotent)
    _reset_source_roads_root(source_root=SOURCE_ROOT2)

    # Run pairing
    from assets_pairing.application.use_case import PairAssetsUseCase
    from assets_pairing.infrastructure.filesystem import (
        FilesystemAssetSource,
        FilesystemSnapshotStore,
    )

    asset_source = FilesystemAssetSource(
        image_dir=SEG_002 / "Images", pcd_dir=SEG_002 / "PCDs")
    snapshot_store = FilesystemSnapshotStore(
        destination_directory_path=SEG_002, move=False)

    summary = PairAssetsUseCase(
        asset_source=asset_source, snapshot_store=snapshot_store).run(start_id=1)

    assert summary.written_pairs >= 1, "Expected at least one paired pothole folder"

    pothole_001 = SEG_002 / "pothole_001"
    assert pothole_001.is_dir(), f"Expected {pothole_001} to exist"
    assert any(p.suffix.lower() in (".jpg", ".png")
               for p in pothole_001.iterdir()), "Missing image in pothole_001"
    assert any(p.suffix.lower() == ".pcd" for p in pothole_001.iterdir()
               ), "Missing pcd in pothole_001"
