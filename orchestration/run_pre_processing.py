from assets.pair_pothole_assets import pair_pothole_assets
import argparse
import os
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-processing for the project.")
    parser.add_argument("--config", help="Path to config file.")
    parser.add_argument("--root-dir", help="Path to root directory.")
    parser.add_argument("--images-folder-name", help="Path to images directory.")
    parser.add_argument("--pcd-folder-name", help="Path to pcd directory.")
    parser.add_argument("--dest-dir", help="Path to destination directory.")
    return parser

def main():
    p = build_parser()
    cfg_path = p.parse_known_args()[0].config
    from common.config import load_config
    cfg = load_config(cfg_path)

    p.set_defaults(
        root_dir=str(Path(cfg.paths.workspace_root) / cfg.paths.source_roads_root),
        images_folder_name=cfg.pre_processing.images_dir,
        pcd_folder_name=cfg.pre_processing.pcd_dir,
    )

    args = p.parse_args()

    root_dir = args.root_dir

    for dirpath, dirnames, _ in os.walk(root_dir):
        if {args.images_folder_name, args.pcd_folder_name}.issubset(set(dirnames)):
            seg = Path(dirpath)
            images_dir = seg / args.images_folder_name
            pcd_dir = seg / args.pcd_folder_name
            dest_dir = seg
            pair_pothole_assets(pcd_dir, images_dir, dest_dir, 1, 3, False, "INFO")


if __name__ == "__main__": 
    main()