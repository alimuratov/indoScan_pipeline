"""Tests for paired_assets_analysis bounded context.

These tests exercise the DDD layering + Strategy pattern setup:
- 4 domain service protocols (preprocess/surface/detect/metrics)
- orchestration via SceneAnalysisService

We keep tests lightweight by using fakes (no Open3D/NumPy).
"""
from __future__ import annotations

import json
import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from paired_assets_analysis.domain.model import (
    DetectionOutcome,
    InspectionStatus,
    MetricsReport,
    PcdPath,
    PointCloud,
    Pothole,
    PotholeMetrics,
    PotholeDetection,
    PreprocessOutcome,
    PreprocessedPointCloud,
    RoadScene,
    SceneId,
    SceneInspection,
    SurfaceEstimation,
    SurfaceModel,
)
from paired_assets_analysis.domain.services import (
    PointCloudPreprocessor,
    PotholeDetector,
    PotholeMetricsEstimator,
    SceneAnalysisService,
    SurfaceEstimator,
)
from paired_assets_analysis.application.use_case import AnalyzePairedAssetsUseCase
from paired_assets_analysis.infrastructure.filesystem import (
    FilesystemRoadSceneSource,
    FilesystemSceneInspectionRepository,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@dataclass
class FakePointCloudPreprocessor(PointCloudPreprocessor):
    outcomes: Dict[str, PreprocessOutcome]

    def preprocess(self, scene: RoadScene) -> PreprocessOutcome:
        return self.outcomes.get(
            str(scene.pcd_path.value),
            PreprocessOutcome(status=InspectionStatus.FAILED),
        )


@dataclass
class FakeSurfaceEstimator(SurfaceEstimator):
    surface: SurfaceModel
    inlier_mask: Any = None

    def estimate(self, preprocessed: PreprocessedPointCloud) -> SurfaceEstimation:
        return SurfaceEstimation(surface=self.surface, inlier_mask=self.inlier_mask)


@dataclass
class FakePotholeDetector(PotholeDetector):
    outcomes_by_raw_count: Dict[int, DetectionOutcome]

    def detect(
        self,
        *,
        preprocessed: PreprocessedPointCloud,
        surface: SurfaceModel,
    ) -> DetectionOutcome:
        return self.outcomes_by_raw_count.get(
            int(preprocessed.raw_point_count),
            DetectionOutcome(status=InspectionStatus.FAILED),
        )


@dataclass
class FakePotholeMetricsEstimator(PotholeMetricsEstimator):
    report: MetricsReport

    def estimate(self, *, detection: PotholeDetection, surface: SurfaceModel) -> MetricsReport:
        return self.report


class FakeSceneInspectionRepository:
    """In-memory repository for testing."""

    def __init__(self) -> None:
        self.saved: List[SceneInspection] = []

    def save(self, *, result: SceneInspection) -> None:
        self.saved.append(result)


class FakeSceneArtifactsRepository:
    """In-memory artifacts repository for testing."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def save_ground_inliers(self, *, scene: RoadScene, source_pcd: PointCloud, inlier_mask: Any) -> None:
        self.calls.append(
            {"scene": scene, "source_pcd": source_pcd, "inlier_mask": inlier_mask}
        )


class FakePotholeInstancesExporter:
    """In-memory pothole instances exporter for testing."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def export(
        self,
        *,
        scene: RoadScene,
        preprocessed: PreprocessedPointCloud,
        detection: PotholeDetection,
        report: MetricsReport,
        surface: SurfaceModel,
    ) -> None:
        self.calls.append(
            {
                "scene": scene,
                "preprocessed": preprocessed,
                "detection": detection,
                "report": report,
                "surface": surface,
            }
        )


@pytest.fixture
def sample_metrics_report() -> MetricsReport:
    """Sample successful metrics report (same shape as pcdtools per_pothole_summary outputs)."""
    return MetricsReport(
        clusters=[
            {
                "points": 100,
                "max_depth": 0.15,
                "mean_depth": 0.08,
                "median_depth": 0.07,
                "hull_area": 0.05,
                "simple_volume": 0.002,
                "delaunay_volume": 0.0018,
            },
            {
                "points": 50,
                "max_depth": 0.10,
                "mean_depth": 0.05,
                "median_depth": 0.04,
                "hull_area": 0.02,
                "simple_volume": 0.0008,
            },
        ],
        overall={
            "max_depth": 0.15,
            "mean_depth": 0.065,
            "median_depth": 0.055,
        },
        aggregate=None,
    )


# -----------------------------------------------------------------------------
# Domain Model Tests
# -----------------------------------------------------------------------------


class TestDomainModel:
    """Tests for domain value objects and entities."""

    def test_scene_id_is_frozen(self):
        scene_id = SceneId(value="pothole_001")
        with pytest.raises(Exception):  # FrozenInstanceError
            scene_id.value = "modified"

    def test_pcd_path_is_frozen(self):
        pcd_path = PcdPath(value=Path("/data/test.pcd"))
        with pytest.raises(Exception):
            pcd_path.value = Path("/other.pcd")

    def test_road_scene_structure(self):
        scene = RoadScene(
            id=SceneId(value="pothole_001"),
            pcd_path=PcdPath(value=Path("/data/pothole_001/scan.pcd")),
        )
        assert scene.id.value == "pothole_001"
        assert scene.pcd_path.value == Path("/data/pothole_001/scan.pcd")
        assert scene.folder == Path("/data/pothole_001")

    def test_pothole_metrics_frozen(self):
        metrics = PotholeMetrics(
            n_points=100,
            max_depth=0.15,
            mean_depth=0.08,
            median_depth=0.07,
            hull_area=0.05,
            volume_convex=0.002,
            volume_delaunay=0.0018,
        )
        assert metrics.n_points == 100
        assert metrics.volume_delaunay == 0.0018

    def test_pothole_metrics_optional_delaunay(self):
        metrics = PotholeMetrics(
            n_points=50,
            max_depth=0.10,
            mean_depth=0.05,
            median_depth=0.04,
            hull_area=0.02,
            volume_convex=0.0008,
        )
        assert metrics.volume_delaunay is None

    def test_surface_model_structure(self):
        surface = SurfaceModel(
            model=[0.0, 0.0, 1.0, -0.5],
            method="ransac",
            metadata={"inliers": 1000},
        )
        assert surface.method == "ransac"
        assert len(surface.model) == 4

    def test_inspection_status_enum(self):
        assert InspectionStatus.OK.value == "ok"
        assert InspectionStatus.EMPTY_CLOUD.value == "empty_cloud"
        assert InspectionStatus.NO_POTHOLE_POINTS.value == "no_pothole_points"


# -----------------------------------------------------------------------------
# Domain Service Tests
# -----------------------------------------------------------------------------


class TestSceneAnalysisService:
    """Tests for SceneAnalysisService domain service."""

    def test_analyze_returns_scene_inspection(self, sample_metrics_report):
        surface = SurfaceModel(
            model=[0.0, 0.0, 1.0, -0.5],
            method="ransac",
            metadata={},
        )
        pre = PreprocessedPointCloud(
            pothole_points=None,
            non_pothole_points=None,
            raw_point_count=123,
        )
        service = SceneAnalysisService(
            preprocessor=FakePointCloudPreprocessor(
                outcomes={
                    "/data/pothole_001/test.pcd": PreprocessOutcome(status=InspectionStatus.OK, data=pre)}
            ),
            surface_estimator=FakeSurfaceEstimator(surface=surface),
            pothole_detector=FakePotholeDetector(
                outcomes_by_raw_count={
                    123: DetectionOutcome(
                        status=InspectionStatus.OK,
                        data=PotholeDetection(
                            filtered_points=None,
                            depths=None,
                            labels=None,
                            n_clusters=2,
                        ),
                    )
                }
            ),
            metrics_estimator=FakePotholeMetricsEstimator(
                report=sample_metrics_report),
        )

        scene = RoadScene(
            id=SceneId(value="pothole_001"),
            pcd_path=PcdPath(value=Path("/data/pothole_001/test.pcd")),
        )
        inspection = service.analyze(scene)

        assert inspection.scene == scene
        assert inspection.status == InspectionStatus.OK
        assert inspection.surface is not None
        assert inspection.surface.method == "ransac"
        assert len(inspection.potholes) == 2

    def test_analyze_maps_potholes_correctly(self, sample_metrics_report):
        surface = SurfaceModel(
            model=[0.0, 0.0, 1.0, -0.5],
            method="ransac",
            metadata={},
        )
        pre = PreprocessedPointCloud(
            pothole_points=None,
            non_pothole_points=None,
            raw_point_count=999,
        )
        service = SceneAnalysisService(
            preprocessor=FakePointCloudPreprocessor(
                outcomes={
                    "/data/pothole_001/test.pcd": PreprocessOutcome(status=InspectionStatus.OK, data=pre)}
            ),
            surface_estimator=FakeSurfaceEstimator(surface=surface),
            pothole_detector=FakePotholeDetector(
                outcomes_by_raw_count={
                    999: DetectionOutcome(
                        status=InspectionStatus.OK,
                        data=PotholeDetection(None, None, None, 2),
                    )
                }
            ),
            metrics_estimator=FakePotholeMetricsEstimator(
                report=sample_metrics_report),
        )

        scene = RoadScene(
            id=SceneId(value="pothole_001"),
            pcd_path=PcdPath(value=Path("/data/pothole_001/test.pcd")),
        )
        inspection = service.analyze(scene)

        # First pothole has delaunay_volume
        assert inspection.potholes[0].cluster_id == 0
        assert inspection.potholes[0].metrics.n_points == 100
        assert inspection.potholes[0].metrics.volume_delaunay == 0.0018

        # Second pothole has no delaunay_volume
        assert inspection.potholes[1].cluster_id == 1
        assert inspection.potholes[1].metrics.n_points == 50
        assert inspection.potholes[1].metrics.volume_delaunay is None

    def test_analyze_handles_failed_status(self):
        surface = SurfaceModel(
            model=[0.0, 0.0, 1.0, -0.5],
            method="ransac",
            metadata={},
        )
        service = SceneAnalysisService(
            preprocessor=FakePointCloudPreprocessor(
                outcomes={
                    "/data/pothole_fail/fail.pcd": PreprocessOutcome(status=InspectionStatus.EMPTY_CLOUD)}
            ),
            surface_estimator=FakeSurfaceEstimator(surface=surface),
            pothole_detector=FakePotholeDetector(outcomes_by_raw_count={}),
            metrics_estimator=FakePotholeMetricsEstimator(
                report=MetricsReport(clusters=[])),
        )

        scene = RoadScene(
            id=SceneId(value="pothole_fail"),
            pcd_path=PcdPath(value=Path("/data/pothole_fail/fail.pcd")),
        )
        inspection = service.analyze(scene)

        assert inspection.status == InspectionStatus.EMPTY_CLOUD
        assert inspection.surface is None
        assert inspection.potholes == []

    def test_analyze_handles_detection_failure_but_keeps_surface(self):
        surface = SurfaceModel(
            model=[0.0, 0.0, 1.0, -0.5],
            method="ransac",
            metadata={},
        )
        pre = PreprocessedPointCloud(
            pothole_points=None,
            non_pothole_points=None,
            raw_point_count=42,
        )
        service = SceneAnalysisService(
            preprocessor=FakePointCloudPreprocessor(
                outcomes={
                    "/data/pothole_noplane/nobelow.pcd": PreprocessOutcome(status=InspectionStatus.OK, data=pre)}
            ),
            surface_estimator=FakeSurfaceEstimator(surface=surface),
            pothole_detector=FakePotholeDetector(
                outcomes_by_raw_count={
                    42: DetectionOutcome(status=InspectionStatus.NO_POINTS_BELOW_PLANE)
                }
            ),
            metrics_estimator=FakePotholeMetricsEstimator(
                report=MetricsReport(clusters=[])),
        )

        scene = RoadScene(
            id=SceneId(value="pothole_noplane"),
            pcd_path=PcdPath(value=Path("/data/pothole_noplane/nobelow.pcd")),
        )
        inspection = service.analyze(scene)

        assert inspection.status == InspectionStatus.NO_POINTS_BELOW_PLANE
        assert inspection.surface is not None

    def test_analyze_calls_artifacts_repo_after_surface_estimation(self):
        surface = SurfaceModel(
            model=[0.0, 0.0, 1.0, -0.5],
            method="patchworkpp+ransac",
            metadata={},
        )
        pre = PreprocessedPointCloud(
            pothole_points=PointCloud(points=[[0.0, 0.0, 0.0]], colors=None),
            non_pothole_points=PointCloud(
                points=[[1.0, 1.0, 0.0]], colors=None),
            raw_point_count=42,
        )
        artifacts_repo = FakeSceneArtifactsRepository()
        mask = object()

        service = SceneAnalysisService(
            preprocessor=FakePointCloudPreprocessor(
                outcomes={
                    "/data/pothole_001/test.pcd": PreprocessOutcome(status=InspectionStatus.OK, data=pre)}
            ),
            surface_estimator=FakeSurfaceEstimator(
                surface=surface, inlier_mask=mask),
            pothole_detector=FakePotholeDetector(
                outcomes_by_raw_count={
                    42: DetectionOutcome(status=InspectionStatus.NO_POINTS_BELOW_PLANE)
                }
            ),
            metrics_estimator=FakePotholeMetricsEstimator(
                report=MetricsReport(clusters=[])),
            artifacts_repo=artifacts_repo,
        )

        scene = RoadScene(
            id=SceneId(value="pothole_001"),
            pcd_path=PcdPath(value=Path("/data/pothole_001/test.pcd")),
        )
        _ = service.analyze(scene)

        assert len(artifacts_repo.calls) == 1
        assert artifacts_repo.calls[0]["scene"] == scene
        assert artifacts_repo.calls[0]["source_pcd"] == pre.non_pothole_points
        assert artifacts_repo.calls[0]["inlier_mask"] is mask

    def test_analyze_calls_pothole_instances_exporter_after_metrics(self, sample_metrics_report):
        surface = SurfaceModel(
            model=[0.0, 0.0, 1.0, -0.5],
            method="patchworkpp+ransac",
            metadata={},
        )
        pre = PreprocessedPointCloud(
            pothole_points=PointCloud(
                points=[[0.0, 0.0, -0.1], [0.1, 0.0, -0.1]], colors=None),
            non_pothole_points=PointCloud(
                points=[[1.0, 1.0, 0.0]], colors=None),
            raw_point_count=42,
        )
        exporter = FakePotholeInstancesExporter()

        service = SceneAnalysisService(
            preprocessor=FakePointCloudPreprocessor(
                outcomes={
                    "/data/pothole_001/test.pcd": PreprocessOutcome(status=InspectionStatus.OK, data=pre)}
            ),
            surface_estimator=FakeSurfaceEstimator(
                surface=surface, inlier_mask=object()),
            pothole_detector=FakePotholeDetector(
                outcomes_by_raw_count={
                    42: DetectionOutcome(
                        status=InspectionStatus.OK,
                        data=PotholeDetection(
                            filtered_points=[
                                [0.0, 0.0, -0.1], [0.1, 0.0, -0.1]],
                            depths=[0.1, 0.1],
                            labels=None,
                            n_clusters=1,
                        ),
                    )
                }
            ),
            metrics_estimator=FakePotholeMetricsEstimator(
                report=sample_metrics_report),
            pothole_exporter=exporter,
        )

        scene = RoadScene(
            id=SceneId(value="pothole_001"),
            pcd_path=PcdPath(value=Path("/data/pothole_001/test.pcd")),
        )
        _ = service.analyze(scene)

        assert len(exporter.calls) == 1
        assert exporter.calls[0]["scene"] == scene
        assert exporter.calls[0]["preprocessed"] == pre
        assert exporter.calls[0]["surface"] == surface


# -----------------------------------------------------------------------------
# Application Use Case Tests
# -----------------------------------------------------------------------------


class FakeRoadSceneSource:
    """Fake scene source for testing."""

    def __init__(self, scenes: List[RoadScene]) -> None:
        self._scenes = scenes

    def list(self, *, folder: Path) -> List[RoadScene]:
        return self._scenes


class TestAnalyzePairedAssetsUseCase:
    """Tests for AnalyzePairedAssetsUseCase application service."""

    def test_run_processes_all_scenes(self, sample_metrics_report):
        scenes = [
            RoadScene(id=SceneId("pothole_001"),
                      pcd_path=PcdPath(Path("/a.pcd"))),
            RoadScene(id=SceneId("pothole_002"),
                      pcd_path=PcdPath(Path("/b.pcd"))),
        ]

        surface = SurfaceModel(
            model=[0.0, 0.0, 1.0, -0.5],
            method="ransac",
            metadata={},
        )
        pre_a = PreprocessedPointCloud(
            pothole_points=None, non_pothole_points=None, raw_point_count=1)
        pre_b = PreprocessedPointCloud(
            pothole_points=None, non_pothole_points=None, raw_point_count=2)

        service = SceneAnalysisService(
            preprocessor=FakePointCloudPreprocessor(
                outcomes={
                    "/a.pcd": PreprocessOutcome(status=InspectionStatus.OK, data=pre_a),
                    "/b.pcd": PreprocessOutcome(status=InspectionStatus.OK, data=pre_b),
                }
            ),
            surface_estimator=FakeSurfaceEstimator(surface=surface),
            pothole_detector=FakePotholeDetector(
                outcomes_by_raw_count={
                    1: DetectionOutcome(status=InspectionStatus.OK, data=PotholeDetection(None, None, None, 2)),
                    2: DetectionOutcome(status=InspectionStatus.OK, data=PotholeDetection(None, None, None, 2)),
                }
            ),
            metrics_estimator=FakePotholeMetricsEstimator(
                report=sample_metrics_report),
        )

        use_case = AnalyzePairedAssetsUseCase(
            scene_source=FakeRoadSceneSource(scenes),
            analysis_service=service,
        )
        results = use_case.run(paired_assets_folder=Path("/data"))

        assert len(results) == 2
        assert results[0].scene.id.value == "pothole_001"
        assert results[1].scene.id.value == "pothole_002"

    def test_run_saves_to_repository_when_provided(self, sample_metrics_report):
        scenes = [
            RoadScene(id=SceneId("pothole_001"),
                      pcd_path=PcdPath(Path("/a.pcd"))),
        ]

        surface = SurfaceModel(
            model=[0.0, 0.0, 1.0, -0.5],
            method="ransac",
            metadata={},
        )
        pre_a = PreprocessedPointCloud(
            pothole_points=None, non_pothole_points=None, raw_point_count=1)
        service = SceneAnalysisService(
            preprocessor=FakePointCloudPreprocessor(
                outcomes={
                    "/a.pcd": PreprocessOutcome(status=InspectionStatus.OK, data=pre_a)}
            ),
            surface_estimator=FakeSurfaceEstimator(surface=surface),
            pothole_detector=FakePotholeDetector(
                outcomes_by_raw_count={
                    1: DetectionOutcome(status=InspectionStatus.OK, data=PotholeDetection(None, None, None, 2))
                }
            ),
            metrics_estimator=FakePotholeMetricsEstimator(
                report=sample_metrics_report),
        )
        repo = FakeSceneInspectionRepository()

        use_case = AnalyzePairedAssetsUseCase(
            scene_source=FakeRoadSceneSource(scenes),
            analysis_service=service,
            inspection_repo=repo,
        )
        use_case.run(paired_assets_folder=Path("/data"))

        assert len(repo.saved) == 1
        assert repo.saved[0].scene.id.value == "pothole_001"

    def test_run_returns_empty_for_no_scenes(self, sample_metrics_report):
        surface = SurfaceModel(
            model=[0.0, 0.0, 1.0, -0.5],
            method="ransac",
            metadata={},
        )
        service = SceneAnalysisService(
            preprocessor=FakePointCloudPreprocessor(outcomes={}),
            surface_estimator=FakeSurfaceEstimator(surface=surface),
            pothole_detector=FakePotholeDetector(outcomes_by_raw_count={}),
            metrics_estimator=FakePotholeMetricsEstimator(
                report=sample_metrics_report),
        )

        use_case = AnalyzePairedAssetsUseCase(
            scene_source=FakeRoadSceneSource([]),
            analysis_service=service,
        )
        results = use_case.run(paired_assets_folder=Path("/data"))

        assert results == []


# -----------------------------------------------------------------------------
# Infrastructure Adapter Tests
# -----------------------------------------------------------------------------


class TestFilesystemRoadSceneSource:
    """Tests for FilesystemRoadSceneSource adapter."""

    def test_list_finds_pothole_folders(self, tmp_path):
        # Setup: create pothole folders with PCDs
        (tmp_path / "pothole_001").mkdir()
        (tmp_path / "pothole_001" / "scan.pcd").touch()

        (tmp_path / "pothole_002").mkdir()
        (tmp_path / "pothole_002" / "data.pcd").touch()

        source = FilesystemRoadSceneSource()
        refs = source.list(folder=tmp_path)

        assert len(refs) == 2
        assert refs[0].id.value == "pothole_001"
        assert refs[1].id.value == "pothole_002"

    def test_list_ignores_non_pothole_folders(self, tmp_path):
        (tmp_path / "pothole_001").mkdir()
        (tmp_path / "pothole_001" / "scan.pcd").touch()

        (tmp_path / "other_folder").mkdir()
        (tmp_path / "other_folder" / "data.pcd").touch()

        source = FilesystemRoadSceneSource()
        refs = source.list(folder=tmp_path)

        assert len(refs) == 1
        assert refs[0].id.value == "pothole_001"

    def test_list_ignores_folders_without_exactly_one_pcd(self, tmp_path):
        # Folder with no PCD
        (tmp_path / "pothole_001").mkdir()
        (tmp_path / "pothole_001" / "image.jpg").touch()

        # Folder with multiple PCDs
        (tmp_path / "pothole_002").mkdir()
        (tmp_path / "pothole_002" / "a.pcd").touch()
        (tmp_path / "pothole_002" / "b.pcd").touch()

        # Valid folder
        (tmp_path / "pothole_003").mkdir()
        (tmp_path / "pothole_003" / "scan.pcd").touch()

        source = FilesystemRoadSceneSource()
        refs = source.list(folder=tmp_path)

        assert len(refs) == 1
        assert refs[0].id.value == "pothole_003"

    def test_list_returns_empty_for_nonexistent_folder(self, tmp_path):
        source = FilesystemRoadSceneSource()
        refs = source.list(folder=tmp_path / "nonexistent")

        assert refs == []

    def test_list_returns_scene_with_folder_derived_from_pcd(self, tmp_path):
        folder = tmp_path / "pothole_001"
        folder.mkdir()
        pcd_file = folder / "scan.pcd"
        pcd_file.touch()

        source = FilesystemRoadSceneSource()
        scenes = source.list(folder=tmp_path)

        assert len(scenes) == 1
        assert scenes[0].id.value == "pothole_001"
        assert scenes[0].pcd_path.value == pcd_file
        assert scenes[0].folder == folder

    def test_custom_prefix(self, tmp_path):
        (tmp_path / "scene_001").mkdir()
        (tmp_path / "scene_001" / "data.pcd").touch()

        (tmp_path / "pothole_001").mkdir()
        (tmp_path / "pothole_001" / "scan.pcd").touch()

        source = FilesystemRoadSceneSource(pothole_prefix="scene_")
        refs = source.list(folder=tmp_path)

        assert len(refs) == 1
        assert refs[0].id.value == "scene_001"


class TestFilesystemSceneInspectionRepository:
    """Tests for FilesystemSceneInspectionRepository adapter."""

    def test_save_creates_json_file(self, tmp_path):
        folder = tmp_path / "pothole_001"
        folder.mkdir()

        inspection = SceneInspection(
            scene=RoadScene(
                id=SceneId("pothole_001"),
                pcd_path=PcdPath(folder / "scan.pcd"),
            ),
            status=InspectionStatus.OK,
            surface=SurfaceModel(
                model=[0.0, 0.0, 1.0, -0.5],
                method="ransac",
                metadata={},
            ),
            potholes=[
                Pothole(
                    cluster_id=0,
                    metrics=PotholeMetrics(
                        n_points=100,
                        max_depth=0.15,
                        mean_depth=0.08,
                        median_depth=0.07,
                        hull_area=0.05,
                        volume_convex=0.002,
                        volume_delaunay=0.0018,
                    ),
                )
            ],
            overall={"max_depth": 0.15},
        )

        repo = FilesystemSceneInspectionRepository()
        repo.save(result=inspection)

        output_file = folder / "inspection_result.json"
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert data["scene_id"] == "pothole_001"
        assert data["status"] == "ok"
        assert data["surface"]["method"] == "ransac"
        assert data["metadata"]["cluster_id"] == 0
        assert data["metadata"]["metrics"]["n_points"] == 100
        assert data["metadata"]["overall"]["max_depth"] == 0.15

    def test_save_with_no_surface(self, tmp_path):
        folder = tmp_path / "pothole_fail"
        folder.mkdir()

        inspection = SceneInspection(
            scene=RoadScene(
                id=SceneId("pothole_fail"),
                pcd_path=PcdPath(folder / "scan.pcd"),
            ),
            status=InspectionStatus.EMPTY_CLOUD,
        )

        repo = FilesystemSceneInspectionRepository()
        repo.save(result=inspection)

        output_file = folder / "inspection_result.json"
        data = json.loads(output_file.read_text())
        assert data["surface"] is None
        assert data["metadata"] == {}

    def test_custom_filename(self, tmp_path):
        folder = tmp_path / "pothole_001"
        folder.mkdir()

        inspection = SceneInspection(
            scene=RoadScene(
                id=SceneId("pothole_001"),
                pcd_path=PcdPath(folder / "scan.pcd"),
            ),
            status=InspectionStatus.OK,
        )

        repo = FilesystemSceneInspectionRepository(
            output_filename="results.json")
        repo.save(result=inspection)

        assert (folder / "results.json").exists()
        assert not (folder / "inspection_result.json").exists()
