#!/bin/bash
#
# IndoScan Pothole Detection Pipeline
# ====================================
# This script runs the complete pipeline from raw assets to final JSON output.
#
# Usage:
#   1. Edit pipeline.config with your paths and parameters
#   2. Run: ./run_pipeline.sh
#
# Or override config via command line:
#   SEGMENT_DIR=/path/to/segment ./run_pipeline.sh
#

set -e  # Exit on error

# find the absolute directory path where this script file lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Load configuration from pipeline.config (if exists)
# =============================================================================

CONFIG_FILE="${SCRIPT_DIR}/pipeline.config"
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from: $CONFIG_FILE"
    source "$CONFIG_FILE"
fi

# =============================================================================
# DEFAULT VALUES (can be overridden by config or environment)
# =============================================================================

SEGMENT_DIR="${SEGMENT_DIR:-}"
IMAGE_DIR="${IMAGE_DIR:-}"
PCD_DIR="${PCD_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/output}"
DATA2_DIR="${DATA2_DIR:-${OUTPUT_ROOT}/data2}"
FINAL_JSON="${FINAL_JSON:-${DATA2_DIR}/trial_output.json}"
EPS="${EPS:-0.05}"
SUMMARY_ONLY="${SUMMARY_ONLY:-true}"
EXPORT_INSTANCES="${EXPORT_INSTANCES:-true}"
FPS="${FPS:-10}"
IMU_INTERVAL="${IMU_INTERVAL:-30.0}"
START_ID="${START_ID:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# =============================================================================
# DO NOT EDIT BELOW THIS LINE
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/scripts/src:${PYTHONPATH}"

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   IndoScan Pothole Detection Pipeline     ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Validate inputs
if [ -z "$SEGMENT_DIR" ]; then
    echo -e "${RED}ERROR: SEGMENT_DIR is not set!${NC}"
    echo "Edit pipeline.config or set SEGMENT_DIR environment variable"
    exit 1
fi

if [ ! -d "$SEGMENT_DIR" ]; then
    echo -e "${RED}ERROR: SEGMENT_DIR does not exist: $SEGMENT_DIR${NC}"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$DATA2_DIR"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Segment dir:      $SEGMENT_DIR"
echo "  Output root:      $OUTPUT_ROOT"
echo "  Data2 dir:        $DATA2_DIR"
echo "  Final JSON:       $FINAL_JSON"
echo "  EPS:              $EPS"
echo "  Summary only:     $SUMMARY_ONLY"
echo "  Export instances: $EXPORT_INSTANCES"
echo "  FPS:              $FPS"
echo "  IMU interval:     $IMU_INTERVAL"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Pair Assets (Optional)
# -----------------------------------------------------------------------------
#
# Default convention: raw assets live under the segment folder:
#   <segment>/Images/ and <segment>/PCDs/
#
# We auto-derive these paths when IMAGE_DIR/PCD_DIR are empty, BUT:
# - Pairing is destructive (it cleans existing pothole_* folders before writing),
#   so in "auto" mode we skip pairing if pothole_* folders already exist.

DERIVED_IMAGE_DIR=""
DERIVED_PCD_DIR=""
if [ -z "$IMAGE_DIR" ] && [ -d "$SEGMENT_DIR/Images" ]; then
    DERIVED_IMAGE_DIR="$SEGMENT_DIR/Images"
fi
if [ -z "$PCD_DIR" ] && [ -d "$SEGMENT_DIR/PCDs" ]; then
    DERIVED_PCD_DIR="$SEGMENT_DIR/PCDs"
fi

PAIR_IMAGE_DIR="${IMAGE_DIR:-$DERIVED_IMAGE_DIR}"
PAIR_PCD_DIR="${PCD_DIR:-$DERIVED_PCD_DIR}"

# Detect existing pothole folders
HAS_POTHOLES="false"
for d in "$SEGMENT_DIR"/pothole_*; do
    if [ -d "$d" ]; then
        HAS_POTHOLES="true"
        break
    fi
done

if [ -n "$PAIR_IMAGE_DIR" ] && [ -n "$PAIR_PCD_DIR" ]; then
    # If we auto-derived (IMAGE_DIR/PCD_DIR were empty) and potholes already exist, skip to avoid wiping.
    if [ "$HAS_POTHOLES" = "true" ] && [ -z "$IMAGE_DIR" ] && [ -z "$PCD_DIR" ]; then
        echo -e "${YELLOW}=== Step 1: Skipped ===${NC}"
        echo "  pothole_* folders already exist; skipping pairing (auto mode)."
        echo "  (Set IMAGE_DIR and PCD_DIR explicitly in pipeline.config to force re-pairing.)"
        echo ""
    else
        echo -e "${GREEN}=== Step 1: Pairing Assets ===${NC}"
        echo "  image_dir: $PAIR_IMAGE_DIR"
        echo "  pcd_dir:   $PAIR_PCD_DIR"

        if [ ! -d "$PAIR_IMAGE_DIR" ]; then
            echo -e "${RED}ERROR: image dir does not exist: $PAIR_IMAGE_DIR${NC}"
            exit 1
        fi
        if [ ! -d "$PAIR_PCD_DIR" ]; then
            echo -e "${RED}ERROR: pcd dir does not exist: $PAIR_PCD_DIR${NC}"
            exit 1
        fi

        python - <<PY
from pathlib import Path

from assets_pairing.application.use_case import PairAssetsUseCase
from assets_pairing.infrastructure.filesystem import FilesystemAssetSource, FilesystemSnapshotStore

image_dir = Path(r"""$PAIR_IMAGE_DIR""")
pcd_dir = Path(r"""$PAIR_PCD_DIR""")
out_dir = Path(r"""$SEGMENT_DIR""")

asset_source = FilesystemAssetSource(image_dir=image_dir, pcd_dir=pcd_dir)
snapshot_store = FilesystemSnapshotStore(destination_directory_path=out_dir, move=False)

summary = PairAssetsUseCase(asset_source=asset_source, snapshot_store=snapshot_store).run(start_id=int($START_ID))

print("Pairing complete.")
print(f"  written_pairs={summary.written_pairs}")
print(f"  matched={len(summary.pairing.matched_keys)}")
print(f"  missing_images={len(summary.pairing.missing_images)}")
print(f"  missing_pcds={len(summary.pairing.missing_pcds)}")
PY

        echo -e "${GREEN}✓ Step 1 complete!${NC}"
        echo ""
    fi
else
    echo -e "${YELLOW}=== Step 1: Skipped ===${NC}"
    echo "  (No pairing inputs found. Set IMAGE_DIR+PCD_DIR, or ensure Images/ and PCDs/ exist.)"
    echo ""
fi

# -----------------------------------------------------------------------------
# Step 2: Paired Assets Analysis
# -----------------------------------------------------------------------------

echo -e "${GREEN}=== Step 2: Analyzing Potholes ===${NC}"
echo "  Running surface estimation, pothole detection, and metrics..."

# Build Python boolean arguments
if [ "$SUMMARY_ONLY" = true ]; then
    PY_SUMMARY="True"
else
    PY_SUMMARY="False"
fi

if [ "$EXPORT_INSTANCES" = true ]; then
    PY_EXPORT="True"
else
    PY_EXPORT="False"
fi

python -c "
import sys
# Workaround for open3d/jupyter comm issue
sys.modules['comm'] = type(sys)('comm')
sys.modules['comm'].create_comm = lambda *args, **kwargs: None

from pathlib import Path
from paired_assets_analysis.entrypoints.paired_assets_analyze import paired_assets_analyze

results = paired_assets_analyze(
    paired_assets_folder=Path('${SEGMENT_DIR}'),
    eps=${EPS},
    summary_only=${PY_SUMMARY},
    persist_results=False,
    export_pothole_instances=${PY_EXPORT},
)

ok_count = sum(1 for r in results if r.status.value == 'ok')
print(f'  Processed {len(results)} scenes (OK: {ok_count}, Failed: {len(results) - ok_count})')
"

echo -e "${GREEN}✓ Step 2 complete!${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Segment Processing + JSON Build
# -----------------------------------------------------------------------------

echo -e "${GREEN}=== Step 3: Processing Segment & Building JSON ===${NC}"

python -m segment_processing.entrypoints.process_segment \
    --segment-dir "$SEGMENT_DIR" \
    --export-data2-root "$DATA2_DIR" \
    --build-json "$FINAL_JSON" \
    --fps "$FPS" \
    --imu-interval "$IMU_INTERVAL" \
    --log-level "$LOG_LEVEL"

echo -e "${GREEN}✓ Step 3 complete!${NC}"
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   ✅ Pipeline Complete!                   ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${GREEN}Output files:${NC}"
echo "  📄 Final JSON:  $FINAL_JSON"
echo "  📁 Data2 tree:  $DATA2_DIR/Roads/"
echo "  📋 Manifest:    $DATA2_DIR/copy_manifest.json"
echo ""

# Show summary of final JSON
if [ -f "$FINAL_JSON" ]; then
    echo -e "${YELLOW}JSON Summary:${NC}"
    python3 -c "
import json
with open('${FINAL_JSON}') as f:
    data = json.load(f)
total_potholes = 0
for road in data.get('roads', []):
    for seg in road.get('road_segments', []):
        total_potholes += len(seg.get('potholes', []))
        print(f\"  Road: {road['id'][:20]}...\")
        print(f\"    Segment: {seg['id'][:20]}...\")
        print(f\"    Length: {seg.get('length_in_km', 0):.3f} km\")
        print(f\"    Potholes: {len(seg.get('potholes', []))}\")
print(f\"  ─────────────────────────\")
print(f\"  Total potholes: {total_potholes}\")
"
fi

echo ""
echo -e "${GREEN}Done! 🎉${NC}"
