import os
import json
import shutil
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_TESTS") != "1",
    reason="Set RUN_REAL_TESTS=1 to run tests that mutate real_test fixtures.",
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TEST_ROOT = REPO_ROOT / "scripts" / "src" / "real_test"
SOURCE_ROOT2 = REAL_TEST_ROOT / "source_roads_root2"
SEG_002 = SOURCE_ROOT2 / "seg-002"
# keep destination under real_test for easy diffing
EXPORT_ROOT = SOURCE_ROOT2 / "data2"


def _reset_source_roads_root(*, source_root: Path, pothole_prefix: str = "pothole_") -> None:
    """Reset a source_roads_root-like directory to a pre-pairing state.

    This test intentionally mutates fixtures on disk (when RUN_REAL_TESTS=1),
    so we keep a small local reset helper to restore the baseline.
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


class _FakeVideoCreator:
    """Writes a dummy output video file and returns a VideoArtifact."""

    def create_video(self, *, images_dir: Path, output_path: Path, fps: int):
        from segment_processing.domain.model import VideoArtifact

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"dummy-mp4")
        return VideoArtifact(path=output_path, fps=fps, frame_count=1, duration_seconds=0.1)


class _FakeRouteLengthCalculator:
    """Writes a dummy route_length.json and returns RouteLengthResult."""

    def calculate(self, *, odometry_file_path: Path, output_path: Path):
        from segment_processing.domain.model import RouteLengthResult

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(
            {"meters": 1000.0, "kilometers": 1.0}, indent=2) + "\n", encoding="utf-8")
        return RouteLengthResult(meters=1000.0, kilometers=1.0, path=output_path)


def _write_min_inspection_result_json(pothole_dir: Path, *, max_depth: float) -> None:
    """Write the minimal inspection_result.json needed by segment_processing + exporter."""
    (pothole_dir / "inspection_result.json").write_text(
        json.dumps(
            {
                "scene_id": pothole_dir.name,
                "status": "ok",
                "metadata": {
                    "cluster_id": 0,
                    "metrics": {
                        "max_depth": max_depth,
                        "hull_area": 0.5,
                        "volume_convex": 0.01,
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_real_segment_processing_exports_data2_under_real_test() -> None:
    """Smoke test: pairing -> (stub) inspection -> segment_processing -> data2 export.

    Notes:
    - Uses fake video + route length adapters to avoid requiring cv2/numpy.
    - Still exercises real filesystem I/O and the data2 exporter.
    """
    if not SEG_002.is_dir():
        pytest.skip(f"Missing fixture: {SEG_002}")

    # Always restore baseline even if the test fails mid-way.
    _reset_source_roads_root(source_root=SOURCE_ROOT2)
    try:
        # 1) Pair assets: Images/ + PCDs/ -> pothole_* folders under the segment dir
        from assets_pairing.application.use_case import PairAssetsUseCase
        from assets_pairing.infrastructure.filesystem import FilesystemAssetSource, FilesystemSnapshotStore

        asset_source = FilesystemAssetSource(
            image_dir=SEG_002 / "Images", pcd_dir=SEG_002 / "PCDs")
        snapshot_store = FilesystemSnapshotStore(
            destination_directory_path=SEG_002, move=False)
        summary = PairAssetsUseCase(
            asset_source=asset_source, snapshot_store=snapshot_store).run(start_id=1)
        assert summary.written_pairs >= 1

        pothole_001 = SEG_002 / "pothole_001"
        pothole_002 = SEG_002 / "pothole_002"
        assert pothole_001.is_dir()
        assert pothole_002.is_dir()

        # 2) Provide inspection_result.json (normally produced by paired_assets_analysis persist step)
        _write_min_inspection_result_json(pothole_001, max_depth=0.123)
        _write_min_inspection_result_json(pothole_002, max_depth=0.045)

        # 3) Segment processing + export under real_test
        from segment_processing.application.use_case import ProcessSegmentConfig, ProcessSegmentUseCase
        from segment_processing.infrastructure.adapters import FilesystemGpsConverter, LegacyDepthTimelineExtractor, LegacyImuProcessor
        from segment_processing.infrastructure.data2_export import Data2SegmentArtifactsRepository
        from segment_processing.infrastructure.workspace import FilesystemSegmentWorkspaceFactory

        use_case = ProcessSegmentUseCase(
            workspace_factory=FilesystemSegmentWorkspaceFactory(),
            video_creator=_FakeVideoCreator(),
            imu_processor=LegacyImuProcessor(),
            route_length_calculator=_FakeRouteLengthCalculator(),
            depth_timeline_extractor=LegacyDepthTimelineExtractor(),
            gps_converter=FilesystemGpsConverter(),
            artifacts_repository=Data2SegmentArtifactsRepository(
                export_root=EXPORT_ROOT),
        )

        result = use_case.run(
            segment_dir=SEG_002,
            config=ProcessSegmentConfig(
                persist_results=True, fps=10, imu_interval_seconds=30.0),
        )

        assert (EXPORT_ROOT / "copy_manifest.json").is_file()
        manifest = json.loads(
            (EXPORT_ROOT / "copy_manifest.json").read_text(encoding="utf-8"))
        seg_entry = manifest["segments"][str(SEG_002.resolve())]
        target_segment_dir = Path(seg_entry["target"])

        # Key outputs exist in the exported target tree
        assert (target_segment_dir / "imu.json").is_file()
        assert (target_segment_dir / "segment_depth_timestamps.json").is_file()
        assert (target_segment_dir / "segment_gps.json").is_file()
        assert (target_segment_dir / "segment_meta.json").is_file()

        # Pothole_meta.json exists for each pothole
        potholes_map = seg_entry["potholes"]
        assert str(pothole_001.resolve()) in potholes_map
        assert str(pothole_002.resolve()) in potholes_map

        pt1_dir = Path(potholes_map[str(pothole_001.resolve())]["target"])
        pt2_dir = Path(potholes_map[str(pothole_002.resolve())]["target"])
        assert (pt1_dir / "pothole_meta.json").is_file()
        assert (pt2_dir / "pothole_meta.json").is_file()

        # Sanity: should not fail completely
        assert result.status.value in ("ok", "partial")
    finally:
        _reset_source_roads_root(source_root=SOURCE_ROOT2)
