import json
import os
import shutil
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_TESTS") != "1",
    reason="Set RUN_REAL_TESTS=1 to run tests that require real .pcd data and Open3D/NumPy",
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_ROOT = REPO_ROOT / "scripts" / \
    "src" / "real_test" / "source_roads_root2" / "seg-002"


def _first_pcd_in(folder: Path) -> Path | None:
    pcds = sorted(p for p in folder.glob("*.pcd") if p.is_file())
    return pcds[0] if pcds else None


def test_real_paired_assets_analysis_smoke() -> None:
    """Smoke test against local real_test data.

    How to run:
        RUN_REAL_TESTS=1 pytest tests/test_real_paired_assets_analysis_smoke.py -s
    """
    if not REAL_TEST_ROOT.is_dir():
        pytest.skip(f"Missing real_test folder: {REAL_TEST_ROOT}")

    # Import only when running (these pull in heavy deps like Open3D/NumPy).
    from paired_assets_analysis.domain.model import InspectionStatus
    from paired_assets_analysis.entrypoints.paired_assets_analyze import paired_assets_analyze

    inspections = paired_assets_analyze(
        paired_assets_folder=REAL_TEST_ROOT,
        summary_only=True,  # keep it lightweight: skip DBSCAN (sklearn)
        persist_results=True,
    )

    assert inspections, f"No scenes found under: {REAL_TEST_ROOT}"

    # Prefer pothole_001 if present.
    inspection = next(
        (i for i in inspections if i.scene.id.value == "pothole_001"),
        inspections[0],
    )

    assert inspection.scene.pcd_path.value.exists(), "PCD path does not exist"
    assert inspection.status != InspectionStatus.FAILED, (
        f"Analysis failed for {inspection.scene.id.value}. status={inspection.status.value}"
    )

    # Helpful debug output under -s
    print(
        "scene=", inspection.scene.id.value,
        "status=", inspection.status.value,
        "potholes=", len(inspection.potholes),
        "overall=", inspection.overall,
    )


@pytest.mark.skip(reason="temporarily disabled")
def test_real_paired_assets_analysis_persists_results(tmp_path: Path) -> None:
    """Verify persist_results writes inspection_result.json, without touching real_test/."""
    pothole_001 = REAL_TEST_ROOT / "pothole_001"
    if not pothole_001.is_dir():
        pytest.skip(f"Missing pothole_001 folder: {pothole_001}")

    pcd = _first_pcd_in(pothole_001)
    if pcd is None:
        pytest.skip(f"No .pcd file found in: {pothole_001}")

    tmp_scene = tmp_path / "pothole_001"
    tmp_scene.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pcd, tmp_scene / pcd.name)

    from paired_assets_analysis.entrypoints.paired_assets_analyze import paired_assets_analyze

    paired_assets_analyze(
        paired_assets_folder=tmp_path,
        summary_only=True,
        persist_results=True,
    )

    out_json = tmp_scene / "inspection_result.json"
    print("inspection_result_json_path=", out_json)
    assert out_json.exists(), f"Expected {out_json} to be created"

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["scene_id"] == "pothole_001"
    assert Path(data["folder"]) == tmp_scene
    assert Path(data["pcd_path"]).name == pcd.name
