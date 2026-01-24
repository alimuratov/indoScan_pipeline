"""Entrypoint: paired_assets_analyze function for orchestrators/CLIs."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from paired_assets_analysis.application.use_case import AnalyzePairedAssetsUseCase
from paired_assets_analysis.domain.model import SceneInspection
from paired_assets_analysis.domain.services import SceneAnalysisService
from paired_assets_analysis.domain.strategies import (
    DefaultPointCloudPreprocessor,
    BackendPatchworkPPSurfaceEstimator,
    DefaultPotholeDetector,
    BackendPotholeMetricsEstimator,
)
from paired_assets_analysis.infrastructure.filesystem import (
    FilesystemRoadSceneSource,
    FilesystemPotholeInstancesExporter,
    FilesystemSceneInspectionRepository,
    FilesystemSceneArtifactsRepository,
)
from paired_assets_analysis.infrastructure.pcdtools_adapter import (
    NumpyPotholeDetectionHandler,
    NumpyPotholeMetricsHandler,
    Open3DIOBackend,
    Open3DPointCloudFormatHandler,
)
from paired_assets_analysis.ports import (
    PotholeInstancesExporter,
    SceneArtifactsRepository,
    SceneInspectionRepository,
)


def paired_assets_analyze(
    *,
    paired_assets_folder: Path,
    inspection_repo: Optional[SceneInspectionRepository] = None,
    artifacts_repo: Optional[SceneArtifactsRepository] = None,
    pothole_exporter: Optional[PotholeInstancesExporter] = None,
    eps: float = 0.1,
    summary_only: bool = False,
    persist_results: bool = False,
    persist_artifacts: bool = False,
    export_pothole_instances: bool = False,
) -> List[SceneInspection]:
    """Entrypoint to analyze a paired assets folder.

    This is the composition root for the paired-assets analysis context.
    It wires up all dependencies and runs the use case.

    Args:
        paired_assets_folder: Path containing pothole_* subfolders.
        inspection_repo: Optional custom repository for persisting results.
        eps: DBSCAN clustering epsilon (default 0.1).
        summary_only: If True, skip clustering (single pothole mode).
        persist_results: If True and no custom repo provided, use filesystem repo.
        persist_artifacts: If True and no custom repo provided, save artifacts under scene folder.
        export_pothole_instances: If True, export one folder per pothole cluster (cropped PCD + copied images + per-cluster JSON).

    Returns:
        List of SceneInspection results for each valid scene.
    """
    # Infrastructure: scene source
    scene_source = FilesystemRoadSceneSource()

    # Infrastructure: adapters (small, focused)
    io_backend = Open3DIOBackend()

    # Surface estimation configuration (hard-coded for now; change here when needed)
    surface_params = {
        # Patchwork++ parameters (names depend on build; unknown params will be ignored)
        "uprightness_thr": 0.95,
        "max_range": 1000.0,
        "num_iter": 20,
    }

    # Format handler runs: Patchwork++ ground extraction -> Open3D RANSAC plane fit
    format_handler = Open3DPointCloudFormatHandler(io_backend=io_backend)
    surface_estimator = BackendPatchworkPPSurfaceEstimator(
        format_handler=format_handler,
        uprightness_thr=float(surface_params.get("uprightness_thr", 0.95)),
        max_range=float(surface_params.get("max_range", 1000.0)),
        num_iter=int(surface_params.get("num_iter", 20)),
    )

    detection_handler = NumpyPotholeDetectionHandler()
    metrics_handler = NumpyPotholeMetricsHandler()

    # Infrastructure: artifacts repository (optional)
    artifacts = artifacts_repo
    if artifacts is None and persist_artifacts:
        artifacts = FilesystemSceneArtifactsRepository()

    exporter = pothole_exporter
    if exporter is None and export_pothole_instances:
        exporter = FilesystemPotholeInstancesExporter()

    # Domain: default strategy implementations (pure, depend on backend port only)
    preprocessor = DefaultPointCloudPreprocessor(io_backend=io_backend)
    pothole_detector = DefaultPotholeDetector(
        detection=detection_handler, eps=eps, summary_only=summary_only
    )
    metrics_estimator = BackendPotholeMetricsEstimator(metrics=metrics_handler)

    # Domain: orchestration service
    analysis_service = SceneAnalysisService(
        preprocessor=preprocessor,
        surface_estimator=surface_estimator,
        pothole_detector=pothole_detector,
        metrics_estimator=metrics_estimator,
        artifacts_repo=artifacts,
        pothole_exporter=exporter,
    )

    # Infrastructure: inspection repository (optional)
    repo = inspection_repo
    if repo is None and persist_results:
        repo = FilesystemSceneInspectionRepository()

    # Application: use case
    use_case = AnalyzePairedAssetsUseCase(
        scene_source=scene_source,
        analysis_service=analysis_service,
        inspection_repo=repo,
    )

    return use_case.run(paired_assets_folder=Path(paired_assets_folder))
