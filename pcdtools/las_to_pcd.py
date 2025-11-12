from __future__ import annotations

"""LAS/LAZ → PCD converter.

Dependencies:
  - laspy (reads .las and .laz; .laz requires lazrs or laszip)
  - open3d (to write PCD)

Usage:
  python -m pcdtools.las_to_pcd INPUT.las OUTPUT.pcd [--keep-scale]

By default colors (if present) are normalized to [0,1]. If there is no color,
intensity (if available) is mapped to grayscale colors. XYZ is written as
float64 to preserve precision.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def _read_las_xyzrgb(path: str):
    import laspy  # type: ignore

    las = laspy.read(path)

    # XYZ as float64 world coordinates
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

    # Try RGB; if not available, fall back to intensity grayscale
    rgb: Optional[np.ndarray] = None
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        rgb = np.vstack([las.red, las.green, las.blue]).T.astype(np.float64)
        # Normalize depending on bit depth (usually 16-bit)
        maxv = 65535.0 if rgb.max() > 255 else 255.0
        if maxv > 0:
            rgb /= maxv
    elif hasattr(las, "intensity"):
        inten = np.asarray(las.intensity, dtype=np.float64)
        maxv = float(inten.max()) if inten.size > 0 else 1.0
        if maxv <= 0:
            maxv = 1.0
        g = (inten / maxv).reshape(-1, 1)
        rgb = np.repeat(g, 3, axis=1)

    return xyz, rgb


def convert_las_to_pcd(las_path: str, pcd_path: str) -> None:
    import open3d as o3d  # type: ignore

    xyz, rgb = _read_las_xyzrgb(las_path)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pc.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0))
    Path(pcd_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(pcd_path, pc)


def main():
    ap = argparse.ArgumentParser(description="Convert LAS/LAZ to PCD (Open3D)")
    ap.add_argument("las", help="Input .las/.laz path")
    ap.add_argument("pcd", help="Output .pcd path")
    args = ap.parse_args()

    convert_las_to_pcd(args.las, args.pcd)
    print(f"Saved PCD -> {args.pcd}")


if __name__ == "__main__":
    main()
