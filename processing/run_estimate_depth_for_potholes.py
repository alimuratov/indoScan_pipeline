#!/usr/bin/env python3
"""run_estimate_depth_for_potholes

Purpose:
- Orchestrate running the pothole parameter estimation script for every pothole
  directory under a given root, writing an ``output.txt`` per pothole.

Inputs (CLI):
- --root: Roads/segment/pothole root to scan (required).
- --estimate-script: Path to the estimator (e.g., ``scripts/processing/estimate_pothole_params.py``) (required).
- --config: Optional config file passed through to the estimator.
- --log-level: Logging level.
- --aggregate-all: Forward-compat flag; may be used by the estimator.

Where/how used:
- Run manually by operators or called from higher-level orchestration to
  precompute pothole metrics and persist them as ``output.txt`` files.

Outputs:
- For each pothole directory detected (contains exactly one ``.pcd`` and at
  least one image), creates/overwrites ``output.txt`` with the estimator's
  stdout. Logs summary and any anomalies.

General workflow:
1) Discover pothole folders under ``--root`` (heuristic: has one ``.pcd`` and an image).
2) For each folder, select the ``.pcd``, run the estimator in that folder as CWD,
   optionally passing ``--config``.
3) Capture stdout and write to ``output.txt`` in the same folder.
"""
import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import List

from common.discovery import discover_pothole_dirs
from common.cli import add_config_arg, add_log_level_arg, parse_args_with_config, setup_logging

from processing.estimate_pothole_params import estimate_pothole_params
from exceptions.exceptions import StepPreconditionError

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def run_estimate_depth(estimate_script: Path, pcd_path: Path, cwd: Path, cfg_path: Path) -> str:
    """Invoke the estimate script with provided extra CLI args and return stdout."""
    cmd = [sys.executable, str(estimate_script), str(pcd_path)]

    if cfg_path:
        cmd.append("--config")
        cmd.append(str(cfg_path))
        
    logging.debug("Running: %s", " ".join(cmd))

    env = os.environ.copy()
    # __file__ → path to run_estimate_depth_for_potholes.py
    # .parent.parent → the scripts root (.../scripts)
    # to compute the scripts root and put it on PYTHONPATH so imports like import pcdtools... work.
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        logging.warning("estimate_depth failed (%d) for %s", proc.returncode, pcd_path)
        if proc.stderr:
            logging.debug("stderr: %s", proc.stderr.strip())
    return (proc.stdout or "").strip()

def run_estimate_depth_for_potholes(root: str, estimate_script_path: Path, cfg_path: Path) -> None:
    root = Path(root).resolve()
    if not root.exists() or not root.is_dir():
        raise StepPreconditionError(
            "ROOT_NOT_DIRECTORY",
            f"Root is not a directory: {root}",
            context="run_estimate_depth_for_potholes",
        )

    estimate_script_path = Path(estimate_script_path).resolve()
    if not estimate_script_path.is_file():
        raise StepPreconditionError(
            "ESTIMATE_SCRIPT_NOT_FOUND",
            f"Estimate script not found at: {estimate_script_path}",
            context="run_estimate_depth_for_potholes",
        )

    pothole_dirs = discover_pothole_dirs(root)
    if not pothole_dirs:
        raise StepPreconditionError(
            "NO_POTHOLE_DIRS_FOUND",
            f"No pothole folders found under: {root}",
            context="run_estimate_depth_for_potholes",
        )

    logging.info("Found %d pothole folder(s)", len(pothole_dirs))

    for pothole_dir in sorted(pothole_dirs):
        # Select a .pcd file (first one)
        pcd_files = sorted([p for p in pothole_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pcd"])
        if not pcd_files:
            raise StepPreconditionError(
                "POTHOLE_PCD_MISSING",
                f"No .pcd in {pothole_dir}",
                context="run_estimate_depth_for_potholes",
            )
        if len(pcd_files) > 1:
            raise StepPreconditionError(
                "MULTIPLE_PCD_FILES",
                f"Multiple .pcd files in {pothole_dir}",
                context="run_estimate_depth_for_potholes",
            )

        pcd_path = pcd_files[0]
        logging.info("Processing: %s", pcd_path)

        # Run estimate_depth and capture output
        output_text = run_estimate_depth(estimate_script_path, pcd_path, cwd=pothole_dir, cfg_path=cfg_path)
        if not output_text:
            raise StepPreconditionError(
                "ESTIMATOR_NO_OUTPUT",
                f"No output produced for {pcd_path}",
                context="run_estimate_depth_for_potholes",
            )

        # Write output.txt in pothole folder
        out_file = pothole_dir / "output.txt"
        try:
            with open(out_file, "w") as f:
                f.write(output_text + ("\n" if output_text and not output_text.endswith("\n") else ""))
            logging.debug("Wrote %s", out_file)
        except Exception:
            raise StepPreconditionError(
                "OUTPUT_WRITE_FAILED",
                f"Failed writing: {out_file}",
                context="run_estimate_depth_for_potholes",
            )

    logging.info("Done. Processed %d pothole folder(s)", len(pothole_dirs))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run estimate script per pothole folder")
    add_config_arg(parser)
    add_log_level_arg(parser)
    parser.add_argument("--root", help="Root directory (roads or a single segment/pothole folder)")
    parser.add_argument("--estimate-script", help="Path to estimate script: estimate_pothole_params.py")
    return parser


def main() -> None:
    def _defaults_from_cfg(cfg):
        return dict(
            workspace_root=cfg.paths.workspace_root,
            root=cfg.paths.source_roads_root,
            scripts_root=cfg.paths.scripts_root,
            log_level=cfg.logging.level,
            estimate_script=cfg.paths.estimate_script,
        )

    args, cfg = parse_args_with_config(build_parser, _defaults_from_cfg)

    setup_logging(args.log_level)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise StepPreconditionError(
            "ROOT_NOT_DIRECTORY",
            f"Root is not a directory: {root}",
            context="main",
        )

    run_estimate_depth_for_potholes(root, os.path.join(args.workspace_root, args.scripts_root, args.estimate_script), args.config)

if __name__ == "__main__":
    main()


