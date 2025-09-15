#!/usr/bin/env python3
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


def run_estimate_depth(estimate_script: Path, pcd_path: Path, cwd: Path, extra_args: List[str]) -> str:
    """Invoke the estimate script with provided extra CLI args and return stdout."""
    cmd = [sys.executable, str(estimate_script), str(pcd_path)] + list(extra_args)
    logging.debug("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        logging.warning("estimate_depth failed (%d) for %s", proc.returncode, pcd_path)
        if proc.stderr:
            logging.debug("stderr: %s", proc.stderr.strip())
    return (proc.stdout or "").strip()


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    """Parse known args for this driver; return (args, passthrough) for estimate script.

    Unknown args are preserved and forwarded to the underlying estimate script, so you can
    add/change flags without editing this file. Common flags are also provided explicitly
    for convenience and clearer discoverability.
    """
    parser = argparse.ArgumentParser(description="Run estimate script per pothole folder")
    parser.add_argument("--root", required=True, help="Root directory (roads or a single segment/pothole folder)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--estimate-script", default=None, help="Path to estimate script (default: estimate_depth.py, fallback: estimate_pothole_params.py)")

    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        logging.error("Root is not a directory: %s", root)
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    estimate_script: Path
    if args.estimate_script:
        estimate_script = Path(args.estimate_script).resolve()
    else:
        estimate_script = script_dir / "estimate_pothole_params.py"
    if not estimate_script.is_file():
        logging.error("Estimate script not found at: %s", estimate_script)
        sys.exit(1)

    pothole_dirs = find_pothole_dirs(root)
    if not pothole_dirs:
        logging.warning("No pothole folders found under: %s", root)
        return

    logging.info("Found %d pothole folder(s)", len(pothole_dirs))

    # Build extra args to forward: pass all unknown args through as-is
    extra: List[str] = list(passthrough)
    # Ensure --aggregate-all is on by default if not provided
    if "--aggregate-all" not in extra:
        extra += ["--aggregate-all"]

    for pothole_dir in sorted(pothole_dirs):
        # Select a .pcd file (first one)
        pcd_files = sorted([p for p in pothole_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pcd"])
        if not pcd_files:
            logging.warning("No .pcd in %s", pothole_dir)
            continue
        if len(pcd_files) > 1:
            logging.debug("Multiple .pcd files in %s; using %s", pothole_dir, pcd_files[0].name)

        pcd_path = pcd_files[0]
        logging.info("Processing: %s", pcd_path)

        # Run estimate_depth and capture output
        output_text = run_estimate_depth(estimate_script, pcd_path, cwd=pothole_dir, extra_args=extra)
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


