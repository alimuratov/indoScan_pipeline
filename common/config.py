import os
import yaml
from dataclasses import dataclass, replace, fields
from typing import Optional


def _read(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Paths:
    workspace_root: str = ""
    source_roads_root: str = ""
    target_roads_root: str = ""
    roads_root: str = ""
    manifest_out: str = "output/data2/copy_manifest.json"
    json_out: str = "output/data2/trial_output.json"
    images_folder_name: str = "raw_images"
    imu_filename: str = "imu.txt"
    gps_filename: str = "gps.txt"
    estimate_script: str = "estimate_pothole_params.py"
    hull_plot_path: Optional[str] = None
    surface_heatmap_path: Optional[str] = None
    scripts_root: str = "scripts"

@dataclass(frozen=True)
class Logging:
    level: str = "INFO"

@dataclass(frozen=True)
class PreProcessing:
    images_dir: str = "Images"
    pcd_dir: str = "PCD"

@dataclass(frozen=True)
class Processing:
    eps: float = 0.1
    eps_max: float = 1_000_000.0
    dbscan_min_samples: int = 10
    aggregate_all: bool = False
    summary_only: bool = False
    visualize_3d: bool = False
    save_surface_heatmap: bool = False
    hull_plot_path: Optional[str] = None
    surface_heatmap_path: Optional[str] = None
    save_hull_2d: bool = False
    scripts_root: str = "scripts"
    process_gps_input_filename: str = "gps.txt"
    estimate_script: str = "estimate_pothole_params.py"
    save_pothole_points_with_fitted_plane: bool = False


@dataclass(frozen=True)
class Media:
    fps: int = 10

@dataclass(frozen=True)
class Copy:
    copy_imu: bool = True
    copy_aggregated_depth: bool = True
    detect_segment_media: bool = True


@dataclass(frozen=True)
class Build:
    include_gps_data: bool = True
    include_depth_data: bool = True


@dataclass(frozen=True)
class Config:
    paths: Paths = Paths()
    logging: Logging = Logging()
    pre_processing: PreProcessing = PreProcessing()
    processing: Processing = Processing()
    copy: Copy = Copy()
    build: Build = Build()
    media: Media = Media()

def load_config(path: Optional[str]) -> Config:
    cfg = Config()
    if path and os.path.isfile(path):
        data = _read(path) or {}
        paths = replace(cfg.paths, **(data.get("paths", {}) or {}))
        logging = replace(cfg.logging, **(data.get("logging", {}) or {}))
        pre_processing = replace(cfg.pre_processing, **(data.get("pre_processing", {}) or {}))
        processing = replace(cfg.processing, **(data.get("processing", {}) or {}))
        copy = replace(cfg.copy, **(data.get("copy", {}) or {}))
        build = replace(cfg.build, **(data.get("build", {}) or {}))
        media = replace(cfg.media, **(data.get("media", {}) or {}))
        cfg = replace(cfg, paths=paths, logging=logging, processing=processing, pre_processing=pre_processing, copy=copy, build=build, media=media)
    return cfg