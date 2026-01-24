# End-to-end pipeline (assets pairing → analysis → segment_processing)

This doc describes the *current* recommended workflow in this repo:

1) **Pair pothole assets** (image + PCD) into `pothole_*` folders under each segment directory  
2) **Analyze paired potholes** (surface estimation + pothole detection + metrics) and write `inspection*.json`  
3) **Run segment_processing** to produce segment-level artifacts (video, GPS, depth timeline, and optional Data2 export)

> All Python modules in this pipeline live under `scripts/src/`.  
> When running from repo root, make sure Python can import them (see Setup below).

## Setup

From repo root:

```bash
export PYTHONPATH="$PWD/scripts/src:${PYTHONPATH:-}"
```

Dependencies (high-level):

- **Pairing (`assets_pairing`)**: pure stdlib
- **Analysis (`paired_assets_analysis`)**:
  - required: `open3d`, `pypatchworkpp`, `numpy`
  - clustering (multi-pothole): `scikit-learn` (DBSCAN)
- **Segment processing (`segment_processing`)**:
  - video: `opencv-python` (`cv2`)
  - rest: stdlib

## 1) Pair assets into `pothole_*` folders

### Expected input (per segment)

```
<segment_dir>/
  Images/
    <timestamp>.jpg
    ...
  PCDs/
    <timestamp>.pcd
    ...
```

Timestamps must match by **stem** (e.g. `1756369609.199511051.jpg` ↔ `1756369609.199511051.pcd`).

### Pair one segment (programmatic)

```bash
python - <<'PY'
from pathlib import Path

from assets_pairing.application.use_case import PairAssetsUseCase
from assets_pairing.infrastructure.filesystem import FilesystemAssetSource, FilesystemSnapshotStore

segment_dir = Path("/path/to/segment_dir")

summary = PairAssetsUseCase(
    asset_source=FilesystemAssetSource(
        image_dir=segment_dir / "Images",
        pcd_dir=segment_dir / "PCDs",
    ),
    snapshot_store=FilesystemSnapshotStore(
        destination_directory_path=segment_dir,
        move=False,  # copy by default
    ),
).run(start_id=1)

print("written_pairs=", summary.written_pairs)
print("missing_images=", len(summary.pairing.missing_images))
print("missing_pcds=", len(summary.pairing.missing_pcds))
PY
```

### Expected output

```
<segment_dir>/
  pothole_001/
    <timestamp>.jpg
    <timestamp>.pcd
  pothole_002/
  ...
```

## 2) Analyze paired potholes (`paired_assets_analysis`)

### Expected input

Point the analyzer at a folder that contains `pothole_*` subfolders, each containing **exactly one** `.pcd`:

```
<paired_assets_folder>/            # typically the segment dir
  pothole_001/
    <timestamp>.pcd
    <timestamp>.jpg
  pothole_002/
  ...
```

### Run analysis on a segment folder

This runs **Patchwork++ ground extraction → Open3D RANSAC plane fit → pothole detection → metrics**.

```bash
python - <<'PY'
from pathlib import Path

from paired_assets_analysis.entrypoints.paired_assets_analyze import paired_assets_analyze

paired_assets_analyze(
    paired_assets_folder=Path("/path/to/segment_dir"),
    summary_only=False,      # IMPORTANT: run DBSCAN clustering
    eps=0.10,                # DBSCAN neighborhood radius (in meters)
    persist_results=True,    # writes inspection_result.json into each pothole folder
    persist_artifacts=True,  # writes ground_inliers.pcd into each pothole folder
    export_pothole_instances=False,  # optional (see below)
)
PY
```

### Outputs (per pothole folder)

- `inspection_result.json` (surface + pothole metrics)
- `ground_inliers.pcd` (if `persist_artifacts=True`)

### Optional: export individual pothole instances

If you set `export_pothole_instances=True`, the pipeline will create sibling folders:

```
pothole_001_c000/
pothole_001_c001/
...
```

Each contains:
- cropped `.pcd` (all scene points inside the cluster bbox)
- copied images
- `inspection_result.json` for that one cluster

## 3) Run `segment_processing`

`segment_processing` produces segment-level artifacts:
- `output_video.mp4` from `raw_images/`
- `imu.json`, `segment_gps.json`, `segment_meta.json`
- `segment_depth_timestamps.json` (reads pothole depths from inspection files)

### Expected input structure

Minimal:

```
<segment_dir>/
  raw_images/
    <timestamp>.jpg
    ...
  gps.txt
  imu.txt
  segment_scan.pcd
  pothole_001/
    <timestamp>.jpg
    <timestamp>.pcd
    inspection_result.json   # produced by paired_assets_analysis (default)
  pothole_002/
    ...
```

Notes:
- `segment_processing` expects `inspection_result.json` (from `paired_assets_analysis`) in each pothole folder.
- It will prefer `metadata.metrics.max_depth` when present, otherwise falls back to `overall.max_depth` and then to the first pothole’s `metrics.max_depth`.

### CLI (one segment)

```bash
python -m segment_processing.entrypoints.process_segment \
  --segment-dir /path/to/segment_dir \
  --fps 10 \
  --imu-interval 30.0 \
  --log-level INFO
```

### Export a Data2 tree + consolidated JSON

```bash
python -m segment_processing.entrypoints.process_segment \
  --segment-dir /path/to/segment_dir \
  --export-data2-root /path/to/indoScan/output/data2 \
  --build-json /path/to/indoScan/output/data2/trial_output.json \
  --fps 10 \
  --imu-interval 30.0 \
  --log-level INFO
```

This will create/update:

```
<export-data2-root>/
  copy_manifest.json
  trial_output.json   # consolidated roads JSON (--build-json)
  Roads/
    rd-<uuid>/
      rs-<uuid>/
        Data/...
        Potholes/pt-<uuid>/...
```

The `--build-json` flag produces the consolidated roads JSON (`trial_output.json`) in a single step — no need to run a separate builder script.

### Export Data2 tree only (without consolidated JSON)

If you only need the Data2 tree:

```bash
python -m segment_processing.entrypoints.process_segment \
  --segment-dir /path/to/segment_dir \
  --export-data2-root /path/to/indoScan/output/data2 \
  --fps 10 \
  --imu-interval 30.0 \
  --log-level INFO
```

## 4) Build consolidated JSON separately (optional)

If you exported the Data2 tree in a previous step and want to rebuild only the JSON:

```bash
./build.sh
```

Repo shortcut (from repo root):

```bash
./build.sh
```

## Debugging / smoke tests (optional)

The repo includes side-effectful smoke tests that run on `scripts/src/real_test/` fixtures.

Run (examples):

```bash
RUN_REAL_TESTS=1 pytest -q tests/test_real_assets_pairing_smoke.py -s
RUN_REAL_TESTS=1 pytest -q tests/test_real_paired_assets_analysis_smoke.py -s
RUN_REAL_TESTS=1 pytest -q tests/test_real_segment_processing_smoke.py -s
```

Note: these smoke tests reset the `real_test` fixtures automatically before/after running.


