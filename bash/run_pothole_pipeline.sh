#!/usr/bin/env bash
set -euo pipefail

# One-off runner for pothole geometry pipeline with hardcoded inputs.
# Just run: scripts/bash/run_pothole_pipeline.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Hardcoded inputs/outputs
INPUT_PCD="$REPO_ROOT/tmp_pytest/test_sample0/example1.pcd"
OUT_DIR="$REPO_ROOT/tin_outputs"

# Behavior toggles
SUMMARY_ONLY=True   # set to False to print multi-cluster summaries
SAVE_HULL=True
SAVE_HEATMAP=True

mkdir -p "$OUT_DIR"

# Ensure local modules are importable
export PYTHONPATH="$REPO_ROOT/scripts:${PYTHONPATH:-}"

python - <<PY
import os, json
from scripts.pcdtools.pipeline import run_geometry_pipeline, analyze_pothole_geometry

pcd_path = r"""${INPUT_PCD}"""
summary_only = ${SUMMARY_ONLY}
save_hull = ${SAVE_HULL}
save_heatmap = ${SAVE_HEATMAP}
out_dir = r"""${OUT_DIR}"""

hull_path = os.path.join(out_dir, "hull.png") if save_hull else None
heatmap_path = os.path.join(out_dir, "heatmap.png") if save_heatmap else None
contrib2d = os.path.join(out_dir, "contrib_2d.png")
contrib3d = os.path.join(out_dir, "contrib_3d_multi.png")
tin_mesh = os.path.join(out_dir, "tin2.ply")
tin_points = os.path.join(out_dir, "tin2points.pcd")
plane_mesh = os.path.join(out_dir, "plane.ply")
complete_mesh = os.path.join(out_dir, "complete_mesh.ply")


# Run the pipeline (no 3D UI)
run_geometry_pipeline(
    pcd_path=pcd_path,
    eps=0.1,
    summary_only=summary_only,
    save_hull_2d=save_hull,
    hull_plot_path=hull_path,
    visualize_3d=False,
    save_surface_heatmap=save_heatmap,
    surface_heatmap_path=heatmap_path,
    save_delaunay_contrib_2d_path=contrib2d,
    save_delaunay_contrib_3d_multi_path=contrib3d,
    save_tin_mesh_path=tin_mesh,
    save_tin_points_path=tin_points,
    tin_z_exaggeration=1.0,
    save_pothole_points_with_fitted_plane=True,
    save_plane_mesh_path=plane_mesh,
    save_complete_pothole_mesh_path=complete_mesh
)

# Write a compact JSON summary for quick inspection
result = analyze_pothole_geometry(pcd_path, summary_only=True)
with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(result, f, indent=2)

print("\nOne-off pipeline run completed.")
print("Input:", pcd_path)
print("Outputs:", out_dir)
print("  - summary.json")
if hull_path: print("  - hull.png")
if heatmap_path: print("  - heatmap.png")
PY

echo "Done. Hint: chmod +x $0 to make it directly executable if needed." 


