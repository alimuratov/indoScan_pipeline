from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, Optional

import numpy as np
import open3d as o3d

# Support running as a script without package context
try:
    from pcdtools.io import read_point_cloud, classify_points_by_color
except Exception:  # pragma: no cover - fallback for direct execution
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from pcdtools.io import read_point_cloud, classify_points_by_color

def remove_pothole_points(pcd_path: str) -> None:


def add_pothole_points(pcd_path: str) -> None:
    

def extract_bbox(
    pcd_path: str,
    *,
    axis_aligned: bool = False,
    red_threshold: float = 0.7,
) -> Optional[Dict[str, Any]]:
    """Extract a single pothole bounding box by red color filtering.

    Assumes exactly one pothole is present and its points are colored red.
    Returns a dict with center, extent, corners (and R for OBB) or None if not found.
    """
    pcd = read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        return None

    pothole_points, _ = classify_points_by_color(pcd, red_threshold=red_threshold)
    if len(pothole_points) == 0:
        return None

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pothole_points)
    box = (
        pc.get_axis_aligned_bounding_box() if axis_aligned else pc.get_oriented_bounding_box()
    )

    entry: Dict[str, Any] = {
        "center": [float(x) for x in box.center],
        "extent": [float(x) for x in box.get_extent()],
        "corners": [[float(x) for x in p] for p in np.asarray(box.get_box_points())],
        "type": "AABB" if axis_aligned else "OBB",
    }
    if not axis_aligned:
        entry["R"] = [[float(x) for x in row] for row in np.asarray(box.R)]
    return entry


def save_bbox_json(
    box: Dict[str, Any],
    out_path: str,
    *,
    pcd_path: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {"box": box}
    if pcd_path is not None:
        payload["pcd_path"] = pcd_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a pothole bounding box from a segment PCD (red-colored pothole)")
    parser.add_argument("pcd_path", help="Path to input PCD/PLY/etc.")
    parser.add_argument(
        "--axis-aligned", action="store_true", help="Use axis-aligned boxes instead of OBB"
    )
    parser.add_argument(
        "--red-threshold", type=float, default=0.7, help="Red channel threshold (default 0.7)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output JSON path; prints to stdout if omitted",
    )
    parser.add_argument(
        "--corners-pcd",
        type=str,
        default=None,
        help="Optional path to save 8 box corners as a PCD",
    )
    args = parser.parse_args()

    box = extract_bbox(
        args.pcd_path,
        axis_aligned=args.axis_aligned,
        red_threshold=args.red_threshold,
    )

    if box is None:
        print(json.dumps({"pcd_path": args.pcd_path, "box": None, "reason": "no red pothole points"}, indent=2))
        return

    if args.output_json:
        save_bbox_json(box, args.output_json, pcd_path=args.pcd_path)
        print(f"Saved box -> {args.output_json}")
    else:
        print(json.dumps({"pcd_path": args.pcd_path, "box": box}, indent=2))

    if args.corners_pcd:
        corners = np.asarray(box["corners"], dtype=float)
        pc_corners = o3d.geometry.PointCloud()
        pc_corners.points = o3d.utility.Vector3dVector(corners)
        o3d.io.write_point_cloud(args.corners_pcd, pc_corners)


if __name__ == "__main__":
    main()