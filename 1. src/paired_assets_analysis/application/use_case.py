"""Application layer: use cases for paired-assets analysis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from paired_assets_analysis.domain.model import SceneInspection
from paired_assets_analysis.domain.services import SceneAnalysisService
from paired_assets_analysis.ports import RoadSceneSource, SceneInspectionRepository


@dataclass(frozen=True)
class AnalyzePairedAssetsUseCase:
    """Use case: analyze all paired pothole folders under a root folder.

    This is an application service that orchestrates:
    - Scene enumeration (via RoadSceneSource port)
    - Analysis (via SceneAnalysisService domain service)
    - Persistence (via optional SceneInspectionRepository port)
    """

    scene_source: RoadSceneSource
    analysis_service: SceneAnalysisService
    inspection_repo: Optional[SceneInspectionRepository] = None

    def run(self, *, paired_assets_folder: Path) -> List[SceneInspection]:
        """Analyze all scenes in the given folder and return inspections.

        Args:
            paired_assets_folder: Path containing pothole_* subfolders.

        Returns:
            List of SceneInspection results for each valid scene.
        """
        inspections: List[SceneInspection] = []

        for ref in self.scene_source.list(folder=paired_assets_folder):
            scene = self.scene_source.load(ref=ref)
            inspection = self.analysis_service.analyze(scene)
            inspections.append(inspection)

            if self.inspection_repo is not None:
                self.inspection_repo.save(result=inspection)

        return inspections
