"""estimate_depth CLI (thin wrapper)

This script delegates pothole depth/area/volume analysis to pcdtools.pipeline.
It only parses CLI arguments and invokes the pipeline with the selected options.
"""

from __future__ import annotations

import argparse

from pcdtools.pipeline import run_geometry_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate pothole depths/areas/volumes from a colored point cloud"
    )
    parser.add_argument("pcd_path", help="Path to input point cloud (PCD/PLY/etc.)")
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="DBSCAN epsilon (meters) for clustering pothole points",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only concise summary and force a single pothole (large eps)",
    )
    parser.add_argument(
        "--save-hull-2d",
        action="store_true",
        help="Save a 2D convex hull plot of pothole points",
    )
    parser.add_argument(
        "--hull-plot-path",
        type=str,
        default=None,
        help="Optional base path for hull plot PNG (cluster suffix gets appended)",
    )
    parser.add_argument(
        "--aggregate-all",
        action="store_true",
        help="Sum areas and volumes across clusters; report average mean depth",
    )
    parser.add_argument(
        "--visualize-3d",
        action="store_true",
        help="Open a 3D viewer with plane and pothole points",
    )
    parser.add_argument(
        "--save-surface-heatmap",
        action="store_true",
        help="Save surface-based depth heatmap (single or per-cluster)",
    )
    parser.add_argument(
        "--surface-heatmap-path",
        type=str,
        default=None,
        help="Base path for depth heatmap PNG (cluster suffix appended)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # If summary-only (and not aggregating), encourage a single cluster by bumping eps
    eps_val = args.eps
    if args.summary_only and not args.aggregate_all and eps_val < 1e6:
        eps_val = 1e6

    run_geometry_pipeline(
        args.pcd_path,
        eps=eps_val,
        summary_only=args.summary_only,
        save_hull_2d=args.save_hull_2d,
        hull_plot_path=args.hull_plot_path,
        aggregate_all=args.aggregate_all,
        visualize_3d=args.visualize_3d,
        save_surface_heatmap=args.save_surface_heatmap,
        surface_heatmap_path=args.surface_heatmap_path,
    )


if __name__ == "__main__":
    main()
