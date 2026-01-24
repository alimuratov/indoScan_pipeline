import os
import shutil
from pathlib import Path

import pytest

# To run:
#   RUN_REAL_TESTS=1 pytest -q tests/test_real_pothole_instances_export_noisy_smoke.py -s


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_TESTS") != "1",
    reason="Set RUN_REAL_TESTS=1 to run tests that require real .pcd data + Patchwork++/Open3D (+ sklearn for DBSCAN clustering)",
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_ROOT = (
    REPO_ROOT
    / "scripts"
    / "src"
    / "real_test"
    / "pothole_export_noisy"
)


def _remove_previous_exports(*, parent: Path, scene_id: str) -> None:
    """Remove previous exporter outputs to keep this smoke test idempotent."""
    for p in sorted(parent.glob(f"{scene_id}_c*")):
        if p.is_dir():
            shutil.rmtree(p)


def test_real_noisy_pothole_exporter_creates_outputs_with_clustering() -> None:
    """Real-data smoke test for exporter on a noisy cloud (summary_only=False)."""
    if not REAL_TEST_ROOT.is_dir():
        pytest.skip(f"Missing real_test folder: {REAL_TEST_ROOT}")

    # Validate deps before executing the pipeline.
    try:
        import open3d  # noqa: F401
    except Exception:
        pytest.skip(
            "Missing dependency `open3d` required for exporter smoke test")

    try:
        import pypatchworkpp  # noqa: F401
    except Exception:
        pytest.skip(
            "Missing dependency `pypatchworkpp` required for exporter smoke test"
        )

    try:
        import sklearn  # noqa: F401
    except Exception:
        pytest.skip(
            "Missing dependency `scikit-learn` required for DBSCAN clustering (summary_only=False)"
        )

    from paired_assets_analysis.domain.model import InspectionStatus
    from paired_assets_analysis.entrypoints.paired_assets_analyze import paired_assets_analyze

    # Expect exactly one input scene folder: pothole_001
    scene_folder = REAL_TEST_ROOT / "pothole_001"
    if not scene_folder.is_dir():
        pytest.skip(f"Missing scene folder: {scene_folder}")

    scene_id = scene_folder.name

    # Ensure a clean slate.
    _remove_previous_exports(parent=REAL_TEST_ROOT, scene_id=scene_id)

    inspections = paired_assets_analyze(
        paired_assets_folder=REAL_TEST_ROOT,
        summary_only=False,  # IMPORTANT: run real clustering (DBSCAN)
        eps=0.1,
        persist_results=True,  # exporter writes per-cluster JSON itself
        persist_artifacts=True,
        export_pothole_instances=True,
    )

    assert inspections, f"No scenes found under: {REAL_TEST_ROOT}"
    assert any(i.status == InspectionStatus.OK for i in inspections), (
        "No OK inspections were produced. "
        "This likely means preprocessing failed or surface/detection/metrics crashed."
    )

    # Assert at least one exported cluster folder exists.
    out_folders = sorted(REAL_TEST_ROOT.glob(f"{scene_id}_c*"))
    assert out_folders, "Exporter did not create any per-cluster folders"

    # Spot-check contents of the first cluster folder.
    out_folder = out_folders[0]

    src_pcds = sorted(scene_folder.glob("*.pcd"))
    assert len(
        src_pcds) == 2, "Test dataset scene folder must contain exactly one .pcd"
    expected_pcd = out_folder / src_pcds[0].name
    assert expected_pcd.is_file(
    ), f"Expected cropped PCD was not created: {expected_pcd}"

    src_imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        src_imgs.extend(scene_folder.glob(ext))
    assert src_imgs, "Test dataset scene folder must contain at least one image"
    for img in src_imgs:
        assert (out_folder /
                img.name).is_file(), f"Expected image copy missing: {img.name}"

    assert (out_folder / "inspection_result.json").is_file()
