"""Filesystem adapters for paired-assets analysis."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from paired_assets_analysis.domain.model import (
    PcdPath,
    RoadScene,
    RoadSceneRef,
    SceneId,
    SceneInspection,
)


# ---------------------------------------------------------------------------
# RoadSceneSource Adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilesystemRoadSceneSource:
    """Filesystem adapter: enumerate pothole_* subfolders as road scenes.

    Each pothole folder is expected to contain exactly one `.pcd` file.
    """

    pothole_prefix: str = "pothole_"

    def list(self, *, folder: Path) -> List[RoadSceneRef]:
        """List all valid pothole scenes in the given folder."""
        folder = Path(folder)
        if not folder.is_dir():
            return []

        scenes: List[RoadSceneRef] = []
        for child in sorted(folder.iterdir()):
            ref = self._try_parse_scene(child)
            if ref is not None:
                scenes.append(ref)
        return scenes

    def load(self, *, ref: RoadSceneRef) -> RoadScene:
        """Load a scene from its reference (lightweight: just metadata)."""
        return RoadScene(id=ref.id, pcd_path=ref.pcd_path, folder=ref.folder)

    def _try_parse_scene(self, child: Path) -> RoadSceneRef | None:
        """Try to parse a directory as a pothole scene; returns None if invalid."""
        if not child.is_dir():
            return None
        if not child.name.startswith(self.pothole_prefix):
            return None

        pcds = self._find_pcd_files(child)
        if len(pcds) != 1:
            return None

        return RoadSceneRef(
            id=SceneId(value=child.name),
            pcd_path=PcdPath(value=pcds[0]),
            folder=child,
        )

    @staticmethod
    def _find_pcd_files(folder: Path) -> List[Path]:
        """Find all .pcd files in a folder."""
        return sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() == ".pcd"
        )


# ---------------------------------------------------------------------------
# SceneInspectionRepository Adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilesystemSceneInspectionRepository:
    """Filesystem adapter: persist SceneInspection as JSON in the scene folder."""

    output_filename: str = "inspection_result.json"

    def save(self, *, result: SceneInspection) -> None:
        """Save inspection result as JSON in the scene's folder."""
        output_path = result.scene.folder / self.output_filename
        output_path.write_text(
            json.dumps(_inspection_to_dict(result), indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Serialization Helpers
# ---------------------------------------------------------------------------


def _inspection_to_dict(inspection: SceneInspection) -> Dict[str, Any]:
    """Convert a SceneInspection to a JSON-serializable dictionary."""
    return {
        "scene_id": inspection.scene.id.value,
        "folder": str(inspection.scene.folder),
        "pcd_path": str(inspection.scene.pcd_path.value),
        "status": inspection.status.value,
        "surface": _surface_to_dict(inspection.surface),
        "potholes": [_pothole_to_dict(p) for p in inspection.potholes],
        "overall": inspection.overall,
    }


def _surface_to_dict(surface) -> Dict[str, Any] | None:
    """Convert a SurfaceModel to a dict, or None if absent."""
    if surface is None:
        return None
    return {
        "model": surface.model,
        "method": surface.method.value,
        "metadata": surface.metadata,
    }


def _pothole_to_dict(pothole) -> Dict[str, Any]:
    """Convert a Pothole to a dict."""
    m = pothole.metrics
    return {
        "cluster_id": pothole.cluster_id,
        "metrics": {
            "n_points": m.n_points,
            "max_depth": m.max_depth,
            "mean_depth": m.mean_depth,
            "median_depth": m.median_depth,
            "hull_area": m.hull_area,
            "volume_convex": m.volume_convex,
            "volume_delaunay": m.volume_delaunay,
        },
    }
