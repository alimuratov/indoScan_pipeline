import os
from pathlib import Path

import pytest

# To run: RUN_REAL_TESTS=1 pytest -q tests/test_real_patchwork_pipeline_smoke.py -s


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_TESTS") != "1",
    reason="Set RUN_REAL_TESTS=1 to run tests that require real .pcd data + Patchwork++/Open3D",
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_ROOT = REPO_ROOT / "scripts" / "src" / \
    "real_test" / "source_roads_root2" / "seg-002"


def test_real_patchwork_pipeline_produces_surface_model() -> None:
    """Integration smoke test for Patchwork++ -> Open3D RANSAC surface estimation.

    This test is intentionally skipped by default because it requires:
    - real .pcd data under scripts/src/real_test/
    - `pypatchworkpp` + `open3d`
    """
    if not REAL_TEST_ROOT.is_dir():
        pytest.skip(f"Missing real_test folder: {REAL_TEST_ROOT}")

    # Validate deps before executing the pipeline.
    try:
        import open3d  # noqa: F401
    except Exception:
        pytest.skip(
            "Missing dependency `open3d` required for real Patchwork++ pipeline test")

    try:
        import pypatchworkpp  # noqa: F401
    except Exception:
        pytest.skip(
            "Missing dependency `pypatchworkpp` required for real Patchwork++ pipeline test")

    from paired_assets_analysis.domain.model import InspectionStatus
    from paired_assets_analysis.entrypoints.paired_assets_analyze import paired_assets_analyze

    inspections = paired_assets_analyze(
        paired_assets_folder=REAL_TEST_ROOT,
        summary_only=True,  # keep it lightweight: skip DBSCAN (sklearn)
        persist_results=False,  # don't write into real_test/
        persist_artifacts=True,
    )

    assert inspections, f"No scenes found under: {REAL_TEST_ROOT}"

    ok = next(
        (i for i in inspections if i.status ==
         InspectionStatus.OK and i.surface is not None),
        None,
    )
    assert ok is not None, (
        "No OK inspections with a surface model were produced. "
        "This likely means preprocessing failed or surface estimation crashed."
    )

    assert ok.surface is not None  # for type checkers
    assert ok.surface.method == "patchworkpp+ransac"
    assert isinstance(ok.surface.model, list) and len(ok.surface.model) == 4
