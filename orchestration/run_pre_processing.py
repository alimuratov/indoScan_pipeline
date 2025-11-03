from assets.pair_pothole_assets import pair_pothole_assets
import argparse
import os
from pathlib import Path
from common.cli import add_config_arg, add_log_level_arg, setup_logging, parse_args_with_config
from validation.validate_input import validate_input
from validation.validate_preprocessed_data import validate_preprocessed_data
from validation.validation_helpers import log_issues
import logging
from typing import List

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-processing for the project.")
    add_config_arg(parser); add_log_level_arg(parser)
    parser.add_argument("--root-dir", help="Path to root directory.")
    parser.add_argument("--images-folder-name", help="Path to images directory.")
    parser.add_argument("--pcd-folder-name", help="Path to pcd directory.")
    parser.add_argument("--dest-dir", help="Path to destination directory.")
    return parser

def main():
    def _defaults_from_cfg(cfg):
        return dict(
            root_dir=str(Path(cfg.paths.workspace_root) / cfg.paths.source_roads_root),
            images_folder_name=cfg.pre_processing.images_dir,
            pcd_folder_name=cfg.pre_processing.pcd_dir,
        )

    args, cfg = parse_args_with_config(build_parser, _defaults_from_cfg)
    setup_logging(args.log_level)

    root_dir = args.root_dir

    for dirpath, dirnames, _ in os.walk(root_dir):
        if {args.images_folder_name, args.pcd_folder_name}.issubset(set(dirnames)):
            seg = Path(dirpath)
            images_dir = seg / args.images_folder_name
            pcd_dir = seg / args.pcd_folder_name
            dest_dir = seg

            issues = validate_input(images_dir, pcd_dir)

            if log_issues(issues, "error"):
                logging.error("❌ Validation failed. Skipping pre-processing for segment %s", seg)
                continue
            else:
                logging.info("✅ Input validation passed. Continuing with pre-processing for segment %s", seg)

            pair_pothole_assets(pcd_dir, images_dir, dest_dir, 1, 3, False, "INFO")

            issues = validate_preprocessed_data(images_dir, seg)

            if log_issues(issues, "error"):
                logging.error("❌ Pre-processing validation failed. Skipping pre-processing for segment %s", seg)
                continue
            else:
                logging.info("✅ Pre-processing validation passed. Continuing with pre-processing for segment %s", seg)

if __name__ == "__main__": 
    main()