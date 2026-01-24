"""Filesystem adapters for paired-assets analysis."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from paired_assets_analysis.domain.model import (
    InspectionStatus,
    MetricsReport,
    PcdPath,
    Pothole,
    PotholeDetection,
    PotholeMetrics,
    PreprocessedPointCloud,
    RoadScene,
    SceneId,
    SceneInspection,
    PointCloud,
    SurfaceModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RoadSceneSource Adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilesystemRoadSceneSource:
    """Filesystem adapter: enumerate pothole_* subfolders as road scenes.

    Each pothole folder is expected to contain exactly one `.pcd` file.
    """

    pothole_prefix: str = "pothole_"

    def list(self, *, folder: Path) -> List[RoadScene]:
        """List all valid pothole scenes in the given folder."""
        folder = Path(folder)
        if not folder.is_dir():
            msg = f"Provided pothole parent folder is not a directory: '{folder}'"
            logger.warning(msg)
            return []

        scenes: List[RoadScene] = []
        for child in sorted(folder.iterdir()):
            scene = self._try_parse_scene(child)
            if scene is not None:
                scenes.append(scene)
        return scenes

    def _try_parse_scene(self, child: Path) -> RoadScene | None:
        """Try to parse a directory as a pothole scene; returns None if invalid."""
        if not child.is_dir():
            if child.name.startswith(self.pothole_prefix):
                msg = f"Expected a directory for a scene folder, but found '{child}'"
                logger.warning(msg)
            return None
        if not child.name.startswith(self.pothole_prefix):
            return None
        # Ignore already-exported cluster folders (e.g. pothole_001_c000).
        # These are treated as *outputs* of analysis, not inputs.
        if self._has_cluster_suffix(child.name):
            return None

        pcds = self._find_pcd_files(child)
        if len(pcds) != 1:
            msg = (
                f"Skipping scene folder '{child}': expected exactly 1 .pcd file, found {len(pcds)}"
            )
            if len(pcds) > 1:
                preview = ", ".join(p.name for p in pcds[:5])
                suffix = "..." if len(pcds) > 5 else ""
                msg = f"{msg} ({preview}{suffix})"
            logger.warning(msg)
            return None

        return RoadScene(
            id=SceneId(value=child.name),
            pcd_path=PcdPath(value=pcds[0]),
        )

    @staticmethod
    def _find_pcd_files(folder: Path) -> List[Path]:
        """Find all .pcd files in a folder."""
        return sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() == ".pcd"
        )

    @staticmethod
    def _has_cluster_suffix(scene_id: str) -> bool:
        """Return True if `scene_id` ends with the exporter suffix `_c###`."""
        if len(scene_id) < 5:
            return False
        suf = scene_id[-5:]
        return suf.startswith("_c") and suf[2:].isdigit()


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


@dataclass(frozen=True)
class FilesystemSceneArtifactsRepository:
    """Filesystem adapter: persist intermediate artifacts under the scene folder."""

    ground_inliers_filename: str = "ground_inliers.pcd"

    def save_ground_inliers(
        self,
        *,
        scene: RoadScene,
        source_pcd: PointCloud,
        inlier_mask: Any,
    ) -> None:
        """Save inlier points as a .pcd under the scene folder."""

        out_path = scene.folder / self.ground_inliers_filename

        # Heavy deps: import only when actually persisting artifacts.
        import numpy as np
        import open3d as o3d

        pts = np.asarray(source_pcd.get_points(), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                "source_pcd.points must be an array-like of shape (N, 3)")

        mask = np.asarray(inlier_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != pts.shape[0]:
            raise ValueError(
                "inlier_mask must be a 1D boolean array aligned with source_pcd.points"
            )

        inlier_pts = pts[mask]

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(inlier_pts)

        cols = source_pcd.get_colors()
        if cols is not None:
            cols_arr = np.asarray(cols, dtype=np.float64)
            if cols_arr.shape == pts.shape:
                inlier_cols = cols_arr[mask]
                if inlier_cols.size > 0 and len(inlier_cols) == len(inlier_pts):
                    o3d_pcd.colors = o3d.utility.Vector3dVector(inlier_cols)

        o3d.io.write_point_cloud(str(out_path), o3d_pcd)


@dataclass(frozen=True)
class FilesystemPotholeInstancesExporter:
    """Filesystem exporter: one folder per pothole cluster, with cropped PCD + copied images + per-cluster JSON."""

    output_filename: str = "inspection_result.json"
    cluster_suffix_fmt: str = "_c{cluster_id:03d}"
    axis_aligned_bbox: bool = False  # legacy default: OBB

    @staticmethod
    def _stack_scene_points_and_colors(
        *,
        preprocessed: PreprocessedPointCloud,
        np_mod: Any,
    ) -> tuple[Any, Any]:
        """Return (scene_pts, scene_cols_or_None) for cropping.

        `scene_pts` contains all scene points (non-pothole + pothole). The ordering is
        not the original file ordering; it is the concatenation order.
        `scene_cols` is stacked in the exact same order when colors are available.
        """
        non_pts = np_mod.asarray(
            preprocessed.non_pothole_points.get_points(), dtype=np_mod.float64
        )
        pot_pts = np_mod.asarray(
            preprocessed.pothole_points.get_points(), dtype=np_mod.float64
        )

        if non_pts.size == 0 and pot_pts.size == 0:
            return np_mod.empty((0, 3), dtype=np_mod.float64), None

        if non_pts.size == 0:
            scene_pts = pot_pts
        elif pot_pts.size == 0:
            scene_pts = non_pts
        else:
            scene_pts = np_mod.vstack([non_pts, pot_pts])

        non_cols = preprocessed.non_pothole_points.get_colors()
        pot_cols = preprocessed.pothole_points.get_colors()
        scene_cols = None
        if non_cols is not None and pot_cols is not None:
            non_cols_arr = np_mod.asarray(non_cols, dtype=np_mod.float64)
            pot_cols_arr = np_mod.asarray(pot_cols, dtype=np_mod.float64)
            if non_cols_arr.shape == non_pts.shape and pot_cols_arr.shape == pot_pts.shape:
                scene_cols = (
                    np_mod.vstack([non_cols_arr, pot_cols_arr])
                    if (non_cols_arr.size and pot_cols_arr.size)
                    else None
                )

        return scene_pts, scene_cols

    @staticmethod
    def _cluster_ids_and_labels(detection: PotholeDetection) -> tuple[list[int], Any]:
        """Return (cluster_ids, labels_or_None) in a consistent way."""
        if detection.labels is None:
            return [0], None
        return list(range(int(detection.n_clusters))), detection.labels

    @staticmethod
    def _list_images(*, folder: Path) -> list[Path]:
        """List image assets to copy for each exported cluster folder."""
        image_files: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_files.extend(sorted(folder.glob(ext)))
        return image_files

    @staticmethod
    def _cluster_points_and_depths(
        *,
        detection: PotholeDetection,
        labels: Any,
        cluster_id: int,
        np_mod: Any,
    ) -> tuple[Any, Any]:
        """Return (cluster_pts, cluster_depths) for bbox computation."""
        if labels is None:
            cluster_pts = np_mod.asarray(
                detection.filtered_points, dtype=np_mod.float64)
            cluster_depths = np_mod.asarray(
                detection.depths, dtype=np_mod.float64)
        else:
            mask = labels == cluster_id
            cluster_pts = np_mod.asarray(
                detection.filtered_points[mask], dtype=np_mod.float64)
            cluster_depths = np_mod.asarray(
                detection.depths[mask], dtype=np_mod.float64)
        return cluster_pts, cluster_depths

    def _bbox_from_cluster_points(self, *, cluster_pts: Any, o3d_mod: Any) -> Any:
        """Compute a bbox (AABB/OBB) from the cluster points (legacy: OBB by default)."""
        pc_cluster = o3d_mod.geometry.PointCloud()
        pc_cluster.points = o3d_mod.utility.Vector3dVector(cluster_pts)
        return (
            pc_cluster.get_axis_aligned_bounding_box()
            if self.axis_aligned_bbox
            else pc_cluster.get_oriented_bounding_box()
        )

    @staticmethod
    def _crop_scene_by_bbox(
        *,
        scene_pts: Any,
        scene_cols: Any,
        box: Any,
        np_mod: Any,
        o3d_mod: Any,
    ) -> tuple[Any, Any]:
        """Crop all scene points inside bbox; return (cropped_pts, cropped_cols_or_None)."""
        idx = box.get_point_indices_within_bounding_box(
            o3d_mod.utility.Vector3dVector(scene_pts)
        )
        idx = np_mod.asarray(idx, dtype=int)
        cropped_pts = (
            scene_pts[idx]
            if idx.size
            else np_mod.empty((0, 3), dtype=np_mod.float64)
        )
        cropped_cols = scene_cols[idx] if (
            scene_cols is not None and idx.size) else None
        return cropped_pts, cropped_cols

    @staticmethod
    def _write_cropped_pcd(
        *,
        out_pcd_path: Path,
        cropped_pts: Any,
        cropped_cols: Any,
        o3d_mod: Any,
    ) -> None:
        out_pcd = o3d_mod.geometry.PointCloud()
        out_pcd.points = o3d_mod.utility.Vector3dVector(cropped_pts)
        if cropped_cols is not None and cropped_cols.shape == cropped_pts.shape and cropped_cols.size:
            out_pcd.colors = o3d_mod.utility.Vector3dVector(cropped_cols)
        o3d_mod.io.write_point_cloud(str(out_pcd_path), out_pcd)

    def _build_cluster_inspection(
        self,
        *,
        new_scene: RoadScene,
        cluster_id: int,
        cluster_pts: Any,
        cluster_depths: Any,
        report: MetricsReport,
        surface: SurfaceModel,
        np_mod: Any,
    ) -> SceneInspection:
        """Build a per-cluster SceneInspection with exactly one pothole record."""
        cluster_metrics = report.clusters[cluster_id] if report.clusters else {
        }
        pothole_metrics = PotholeMetrics(
            n_points=int(cluster_metrics.get("points", len(cluster_pts))),
            max_depth=float(
                cluster_metrics.get(
                    "max_depth",
                    float(cluster_depths.max()) if cluster_depths.size else 0.0,
                )
            ),
            mean_depth=float(
                cluster_metrics.get(
                    "mean_depth",
                    float(cluster_depths.mean()) if cluster_depths.size else 0.0,
                )
            ),
            median_depth=float(
                cluster_metrics.get(
                    "median_depth",
                    float(np_mod.median(cluster_depths)
                          ) if cluster_depths.size else 0.0,
                )
            ),
            hull_area=float(cluster_metrics.get("hull_area", 0.0)),
            volume_convex=float(cluster_metrics.get("simple_volume", 0.0)),
            volume_delaunay=(
                float(cluster_metrics["delaunay_volume"])
                if "delaunay_volume" in cluster_metrics
                else None
            ),
        )
        pothole = Pothole(cluster_id=int(cluster_id), metrics=pothole_metrics)
        return SceneInspection(
            scene=new_scene,
            status=InspectionStatus.OK,
            surface=surface,
            potholes=[pothole],
            overall=None,
        )

    def export(
        self,
        *,
        scene: RoadScene,
        preprocessed: PreprocessedPointCloud,
        detection: PotholeDetection,
        report: MetricsReport,
        surface: SurfaceModel,
    ) -> None:
        # Heavy deps: import only when actually exporting.
        import numpy as np
        import open3d as o3d

        src_folder = scene.folder
        dst_parent = src_folder.parent

        # Load full scene points (non-pothole + pothole) for cropping “all scene points inside bbox”
        scene_pts, scene_cols = self._stack_scene_points_and_colors(
            preprocessed=preprocessed, np_mod=np
        )
        if np.asarray(scene_pts).size == 0:
            return

        cluster_ids, labels = self._cluster_ids_and_labels(detection)
        image_files = self._list_images(folder=src_folder)

        src_pcd_name = scene.pcd_path.value.name

        exported_cluster_ids: list[int] = []
        for cluster_id in cluster_ids:
            cluster_pts, cluster_depths = self._cluster_points_and_depths(
                detection=detection, labels=labels, cluster_id=int(cluster_id), np_mod=np
            )

            if cluster_pts.size == 0:
                continue

            box = self._bbox_from_cluster_points(
                cluster_pts=cluster_pts, o3d_mod=o3d)

            cropped_pts, cropped_cols = self._crop_scene_by_bbox(
                scene_pts=scene_pts,
                scene_cols=scene_cols,
                box=box,
                np_mod=np,
                o3d_mod=o3d,
            )

            # Create new folder name
            suffix = self.cluster_suffix_fmt.format(cluster_id=int(cluster_id))
            new_scene_id = f"{scene.id.value}{suffix}"
            out_folder = dst_parent / new_scene_id
            out_folder.mkdir(parents=True, exist_ok=True)

            # Copy images
            for img in image_files:
                shutil.copy2(img, out_folder / img.name)

            # Write cropped PCD
            out_pcd_path = out_folder / src_pcd_name
            self._write_cropped_pcd(
                out_pcd_path=out_pcd_path,
                cropped_pts=cropped_pts,
                cropped_cols=cropped_cols,
                o3d_mod=o3d,
            )

            new_scene = RoadScene(
                id=SceneId(value=new_scene_id),
                pcd_path=PcdPath(value=out_pcd_path),
            )
            inspection = self._build_cluster_inspection(
                new_scene=new_scene,
                cluster_id=int(cluster_id),
                cluster_pts=cluster_pts,
                cluster_depths=cluster_depths,
                report=report,
                surface=surface,
                np_mod=np,
            )

            FilesystemSceneInspectionRepository(output_filename=self.output_filename).save(
                result=inspection
            )
            exported_cluster_ids.append(int(cluster_id))

        # Cleanup: delete the original scene folder once all clusters were exported.
        # This is the default intended behavior: after export, cluster folders become the new units.
        # Note: cluster folders are ignored as inputs by FilesystemRoadSceneSource.
        if cluster_ids and len(exported_cluster_ids) == len(cluster_ids) and src_folder.is_dir():
            shutil.rmtree(src_folder)


# ---------------------------------------------------------------------------
# Serialization Helpers
# ---------------------------------------------------------------------------


def _inspection_to_dict(inspection: SceneInspection) -> Dict[str, Any]:
    """Convert a SceneInspection to a JSON-serializable dictionary."""
    clusters = [_pothole_to_dict(p) for p in inspection.potholes]

    metadata: Dict[str, Any] = {}
    if len(clusters) == 1:
        metadata = {
            "cluster_id": clusters[0].get("cluster_id"),
            "metrics": clusters[0].get("metrics"),
        }
    elif len(clusters) > 1:
        metadata = {
            "clusters": clusters,
            "n_clusters": len(clusters),
        }

    if inspection.overall is not None:
        metadata["overall"] = inspection.overall

    payload: Dict[str, Any] = {
        "scene_id": inspection.scene.id.value,
        "status": inspection.status.value,
        "surface": _surface_to_dict(inspection.surface),
        "metadata": metadata,
    }
    return payload


def _surface_to_dict(surface) -> Dict[str, Any] | None:
    """Convert a SurfaceModel to a dict, or None if absent."""
    if surface is None:
        return None
    return {
        "model": surface.model,
        "method": surface.method,
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
