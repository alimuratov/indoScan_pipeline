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


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def is_pothole_dir(directory: Path) -> bool:
    """Heuristic: a pothole folder contains at least one .pcd and one image."""
    try:
        names = {p.name.lower() for p in directory.iterdir() if p.is_file()}
    except Exception:
        return False
    has_pcd = any(n.endswith(".pcd") for n in names)
    has_img = any(any(n.endswith(ext) for ext in IMAGE_EXTS) for n in names)
    return has_pcd and has_img


def find_pothole_dirs(root: Path) -> List[Path]:
    """Return all pothole directories under root (or root itself if matches)."""
    # If root itself is a pothole dir, return it; else walk recursively
    if is_pothole_dir(root):
        return [root]
    out: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        d = Path(dirpath)
        if is_pothole_dir(d):
            out.append(d)
    return out


def run_estimate_depth(estimate_script: Path, pcd_path: Path, cwd: Path, cfg_path: Path) -> str:
    """Invoke the estimate script with provided extra CLI args and return stdout."""
    cmd = [sys.executable, str(estimate_script), str(pcd_path)]

    if cfg_path:
        cmd.append("--config")
        cmd.append(str(cfg_path))
        
    logging.debug("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        logging.warning("estimate_depth failed (%d) for %s", proc.returncode, pcd_path)
        if proc.stderr:
            logging.debug("stderr: %s", proc.stderr.strip())
    return (proc.stdout or "").strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run estimate script per pothole folder")
    parser.add_argument("--config", help="Path to config file.")
    parser.add_argument("--root", required=True, help="Root directory (roads or a single segment/pothole folder)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--estimate-script", default=None, help="Path to estimate script: estimate_pothole_params.py)", required=True)
    return parser


def main() -> None:
    p = build_parser()
    cfg_path = p.parse_known_args()[0].config

    from common.config import load_config
    cfg = load_config(cfg_path)

    p.set_defaults(
        root=cfg.paths.source_roads_root,
        log_level=cfg.logging.level,
        estimate_script=cfg.paths.estimate_script,
    )

    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        logging.error("Root is not a directory: %s", root)
        sys.exit(1)


    estimate_script = Path(args.estimate_script).resolve()
    if not estimate_script.is_file():
        logging.error("Estimate script not found at: %s", estimate_script)
        sys.exit(1)

    pothole_dirs = find_pothole_dirs(root)
    if not pothole_dirs:
        logging.warning("No pothole folders found under: %s", root)
        return

    logging.info("Found %d pothole folder(s)", len(pothole_dirs))

    for pothole_dir in sorted(pothole_dirs):
        # Select a .pcd file (first one)
        pcd_files = sorted([p for p in pothole_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pcd"])
        if not pcd_files:
            logging.warning("No .pcd in %s", pothole_dir)
            continue
        if len(pcd_files) > 1:
            # Log error and continue
            logging.error("Multiple .pcd files in %s; using %s", pothole_dir, pcd_files[0].name)
            continue

        pcd_path = pcd_files[0]
        logging.info("Processing: %s", pcd_path)

        # Run estimate_depth and capture output
        output_text = run_estimate_depth(estimate_script, pcd_path, cwd=pothole_dir, cfg_path=cfg_path)
        if not output_text:
            logging.warning("No output produced for %s", pcd_path)

        # Write output.txt in pothole folder
        out_file = pothole_dir / "output.txt"
        try:
            with open(out_file, "w") as f:
                f.write(output_text + ("\n" if output_text and not output_text.endswith("\n") else ""))
            logging.info("Wrote %s", out_file)
        except Exception:
            logging.exception("Failed writing: %s", out_file)

    logging.info("Done. Processed %d pothole folder(s)", len(pothole_dirs))


if __name__ == "__main__":
    main()


