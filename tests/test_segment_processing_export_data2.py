import json
from pathlib import Path

import pytest


class _FakeVideoCreator:
    """Test adapter: writes a dummy mp4 file and returns a VideoArtifact."""

    def create_video(self, *, images_dir: Path, output_path: Path, fps: int):
        from segment_processing.domain.model import VideoArtifact

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"dummy-mp4")
        return VideoArtifact(path=output_path, fps=fps, frame_count=1, duration_seconds=0.1)


class _FakeRouteLengthCalculator:
    """Test adapter: writes a dummy route_length.json and returns RouteLengthResult."""

    def calculate(self, *, odometry_file_path: Path, output_path: Path):
        from segment_processing.domain.model import RouteLengthResult

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"meters": 1000.0, "kilometers": 1.0}, indent=2) + "\n",
            encoding="utf-8",
        )
        return RouteLengthResult(meters=1000.0, kilometers=1.0, path=output_path)


def _write_dummy_jpg(path: Path) -> None:
    # The pipeline only depends on filenames existing for timestamp parsing in tests.
    path.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG SOI/EOI markers


def test_segment_processing_exports_legacy_data2_tree(tmp_path: Path) -> None:
    # Arrange: minimal source segment structure
    segment_dir = tmp_path / "source_roads_root" / "roadA" / "seg-001"
    segment_dir.mkdir(parents=True, exist_ok=True)

    raw_images = segment_dir / "raw_images"
    raw_images.mkdir(parents=True, exist_ok=True)
    _write_dummy_jpg(raw_images / "1756369601.000000000.jpg")  # t0
    _write_dummy_jpg(raw_images / "1756369609.000000000.jpg")

    # GPS
    (segment_dir / "gps.txt").write_text(
        "\n".join(
            [
                "#timestamp lat lng alt",
                "1756369601.0 -6.0 106.0 50.0",
                "1756369602.0 -6.1 106.1 51.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # IMU / odometry
    (segment_dir / "imu.txt").write_text(
        "\n".join(
            [
                "#timestamp x y z",
                "1756369601.0 0 0 -0.01",
                "1756369631.0 0 0 -0.02",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Segment lidar scan (copied to Data/Lidar Scan/)
    (segment_dir / "segment_scan.pcd").write_text("VERSION .7\n", encoding="utf-8")

    # One pothole folder with inspection_result.json + media
    pothole_dir = segment_dir / "pothole_001"
    pothole_dir.mkdir(parents=True, exist_ok=True)
    _write_dummy_jpg(pothole_dir / "1756369609.000000000.jpg")
    (pothole_dir / "1756369609.000000000.pcd").write_text("VERSION .7\n", encoding="utf-8")

    (pothole_dir / "inspection_result.json").write_text(
        json.dumps(
            {
                "scene_id": "pothole_001",
                "status": "ok",
                "metadata": {
                        "cluster_id": 0,
                        "metrics": {
                            "max_depth": 0.123,
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

    export_root = tmp_path / "data2"

    # Wire use case with fake adapters (no cv2/numpy needed)
    from segment_processing.application.use_case import ProcessSegmentConfig, ProcessSegmentUseCase
    from segment_processing.infrastructure.adapters import (
        FilesystemGpsConverter,
        LegacyDepthTimelineExtractor,
        LegacyImuProcessor,
    )
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
            export_root=export_root),
    )

    # Act
    result = use_case.run(
        segment_dir=segment_dir,
        config=ProcessSegmentConfig(persist_results=True),
    )

    # Assert: manifest + target tree exist
    assert (export_root / "copy_manifest.json").is_file()
    manifest = json.loads(
        (export_root / "copy_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("version") == 2
    assert "segments" in manifest and isinstance(manifest["segments"], dict)

    seg_entry = manifest["segments"][str(segment_dir.resolve())]

    target_segment_dir = Path(seg_entry["target"])
    assert target_segment_dir.is_dir()

    # Segment JSON outputs
    assert (target_segment_dir / "imu.json").is_file()
    assert (target_segment_dir / "segment_depth_timestamps.json").is_file()
    assert (target_segment_dir / "segment_gps.json").is_file()
    assert (target_segment_dir / "segment_meta.json").is_file()

    # Segment media outputs
    assert (target_segment_dir / "Data" /
            "Survey video" / "output_video.mp4").is_file()
    assert (target_segment_dir / "Data" /
            "Lidar Scan" / "segment_scan.pcd").is_file()

    # Pothole outputs
    assert seg_entry["potholes"], "Expected at least one pothole entry in manifest"
    pothole_entry = seg_entry["potholes"][str(pothole_dir.resolve())]
    target_pothole_dir = Path(pothole_entry["target"])
    assert (target_pothole_dir / "pothole_meta.json").is_file()
    assert (target_pothole_dir / "Image" /
            "1756369609.000000000.jpg").is_file()
    assert (target_pothole_dir / "Lidar Scan" /
            "1756369609.000000000.pcd").is_file()

    meta = json.loads(
        (target_pothole_dir / "pothole_meta.json").read_text(encoding="utf-8"))
    assert meta["depth"] == pytest.approx(0.123)
    assert meta["area"] == pytest.approx(0.5)
    assert meta["volume"] == pytest.approx(0.01)

    # Sanity: export shouldn't change processing result semantics
    assert result.segment.id.value == "seg-001"
