"""Entrypoint: paired_assets_analyze function for orchestrators/CLIs."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from paired_assets_analysis.application.use_case import AnalyzePairedAssetsUseCase
from paired_assets_analysis.domain.model import SceneInspection
from paired_assets_analysis.domain.services import SceneAnalysisService
from paired_assets_analysis.domain.strategies import (
    BackendPointCloudPreprocessor,
    BackendRansacSurfaceEstimator,
    BackendPotholeDetector,
    BackendPotholeMetricsEstimator,
)
from paired_assets_analysis.infrastructure.filesystem import (
    FilesystemRoadSceneSource,
    FilesystemSceneInspectionRepository,
)
from paired_assets_analysis.infrastructure.pcdtools_adapter import PcdToolsBackend
from paired_assets_analysis.ports import SceneInspectionRepository


def paired_assets_analyze(
    *,
    paired_assets_folder: Path,
    inspection_repo: Optional[SceneInspectionRepository] = None,
    eps: float = 0.1,
    summary_only: bool = False,
    persist_results: bool = False,
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

    Returns:
        List of SceneInspection results for each valid scene.
    """
    # Infrastructure: scene source
    scene_source = FilesystemRoadSceneSource()

    # Infrastructure: processing backend (libraries + IO)
    backend = PcdToolsBackend()

    # Domain: default strategy implementations (pure, depend on backend port only)
    preprocessor = BackendPointCloudPreprocessor(backend=backend)
    surface_estimator = BackendRansacSurfaceEstimator(backend=backend)
    pothole_detector = BackendPotholeDetector(
        backend=backend, eps=eps, summary_only=summary_only
    )
    metrics_estimator = BackendPotholeMetricsEstimator(backend=backend)

    # Domain: orchestration service
    analysis_service = SceneAnalysisService(
        preprocessor=preprocessor,
        surface_estimator=surface_estimator,
        pothole_detector=pothole_detector,
        metrics_estimator=metrics_estimator,
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
