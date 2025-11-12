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
from scripts.pcdtools.pipeline import PipelineBuilder

pcd_path = r"""${INPUT_PCD}"""
summary_only = ${SUMMARY_ONLY}
save_hull = ${SAVE_HULL}
save_heatmap = ${SAVE_HEATMAP}
out_dir = r"""${OUT_DIR}"""

# Build and configure pipeline using fluent API
builder = PipelineBuilder(pcd_path)

# Configure clustering
if summary_only:
    builder.without_clustering()
else:
    builder.with_clustering(eps=0.1)

# Configure outputs
builder.with_tin_mesh(os.path.join(out_dir, "tin2.ply")) \\
    .with_tin_points(os.path.join(out_dir, "tin2points.pcd")) \\
    .with_plane_mesh(os.path.join(out_dir, "plane.ply")) \\
    .with_complete_mesh(os.path.join(out_dir, "complete_mesh2.ply")) \\
    .with_plane_with_hole(os.path.join(out_dir, "plane_with_hole2.ply")) \\
    .with_combined_mesh(os.path.join(out_dir, "combined_mesh2.ply")) \\
    .with_overlay_mesh(os.path.join(out_dir, "overlay_with_pcd2.ply")) \\
    .with_delaunay_contrib_2d(os.path.join(out_dir, "contrib_2d.png")) \\
    .with_delaunay_contrib_3d(os.path.join(out_dir, "contrib_3d_multi.png")) \\
    .with_pothole_plane_helper() \\
    .with_pcd_sphere_params(radius=0.02, max_points=8000, resolution=3, seed=0)

# Conditionally add hull and heatmap
if save_hull:
    builder.with_hull_plot(os.path.join(out_dir, "hull.png"))
if save_heatmap:
    builder.with_surface_heatmap(os.path.join(out_dir, "heatmap.png"))

# Run the pipeline
builder.run()

# Write a compact JSON summary for quick inspection
result = builder.analyze()
with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump(result, f, indent=2)

print("\\nOne-off pipeline run completed.")
print("Input:", pcd_path)
print("Outputs:", out_dir)
print("  - summary.json")
if save_hull: print("  - hull.png")
if save_heatmap: print("  - heatmap.png")
PY

echo "Done. Hint: chmod +x $0 to make it directly executable if needed." 


