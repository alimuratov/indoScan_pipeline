import os
import shutil
from pathlib import Path

import pytest

# To run:
#   RUN_REAL_TESTS=1 pytest -q tests/test_real_pothole_instances_export_smoke.py -s


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_TESTS") != "1",
    reason="Set RUN_REAL_TESTS=1 to run tests that require real .pcd data + Patchwork++/Open3D",
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_ROOT = (
    REPO_ROOT
    / "scripts"
    / "src"
    / "real_test"
    / "pothole_instances_export_smoke"
)


def _remove_previous_exports(*, parent: Path, scene_id: str) -> None:
    """Remove previous exporter outputs to keep this smoke test idempotent.

    Export folders are siblings of the input scene folder:
      <parent>/<scene_id>_c000, <parent>/<scene_id>_c001, ...
    """
    for p in sorted(parent.glob(f"{scene_id}_c*")):
        if p.is_dir():
            shutil.rmtree(p)


def test_real_pothole_instances_export_creates_cluster_folders() -> None:
    """Integration smoke test for pothole instance export feature.

    Asserts that running the pipeline with `export_pothole_instances=True` creates:
    - one per-cluster folder `<scene_id>_c000` under the same parent folder
    - a cropped .pcd with the same name as the source .pcd
    - a copied image file
    - a per-cluster `inspection_result.json`
    """
    if not REAL_TEST_ROOT.is_dir():
        pytest.skip(f"Missing real_test folder: {REAL_TEST_ROOT}")

    # Validate deps before executing the pipeline.
    try:
        import open3d  # noqa: F401
    except Exception:
        pytest.skip(
            "Missing dependency `open3d` required for real pothole exporter smoke test"
        )

    try:
        import pypatchworkpp  # noqa: F401
    except Exception:
        pytest.skip(
            "Missing dependency `pypatchworkpp` required for real pothole exporter smoke test"
        )

    from paired_assets_analysis.domain.model import InspectionStatus
    from paired_assets_analysis.entrypoints.paired_assets_analyze import paired_assets_analyze

    # This dataset contains exactly one input scene folder.
    scene_dirs = sorted(
        p for p in REAL_TEST_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("pothole_")
    )
    assert scene_dirs, f"No scene folders found under: {REAL_TEST_ROOT}"
    assert len(scene_dirs) == 1, (
        f"Expected exactly 1 scene folder under {REAL_TEST_ROOT}, found {len(scene_dirs)}: "
        + ", ".join(p.name for p in scene_dirs)
    )
    scene_folder = scene_dirs[0]
    scene_id = scene_folder.name

    # Ensure a clean slate so the source folder isn't mixed with previously exported outputs.
    _remove_previous_exports(parent=REAL_TEST_ROOT, scene_id=scene_id)

    inspections = paired_assets_analyze(
        paired_assets_folder=REAL_TEST_ROOT,
        summary_only=False,  # keep it lightweight: skip DBSCAN (sklearn)
        persist_results=False,  # exporter writes per-cluster JSON itself
        persist_artifacts=False,
        export_pothole_instances=True,
        eps=0.1
    )

    assert inspections, f"No scenes found under: {REAL_TEST_ROOT}"
    assert any(i.status == InspectionStatus.OK for i in inspections), (
        "No OK inspections were produced. "
        "This likely means preprocessing failed or surface/detection/metrics crashed."
    )

    # Expected output: one cluster folder in summary_only mode.
    out_folder = REAL_TEST_ROOT / f"{scene_id}_c000"
    assert out_folder.is_dir(
    ), f"Expected exporter output folder was not created: {out_folder}"

    # The exporter writes the cropped cloud under the same filename as the source .pcd.
    src_pcds = sorted(scene_folder.glob("*.pcd"))
    assert len(
        src_pcds) == 1, "Test dataset scene folder must contain exactly one .pcd"
    expected_pcd = out_folder / src_pcds[0].name
    assert expected_pcd.is_file(
    ), f"Expected cropped PCD was not created: {expected_pcd}"

    # Images are copied verbatim from the source folder.
    src_imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        src_imgs.extend(scene_folder.glob(ext))
    assert src_imgs, "Test dataset scene folder must contain at least one image"
    for img in src_imgs:
        assert (out_folder /
                img.name).is_file(), f"Expected image copy missing: {img.name}"

    # Per-cluster inspection JSON
    assert (out_folder / "inspection_result.json").is_file()
