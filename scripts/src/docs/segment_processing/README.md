# segment_processing (bounded context)

Process a source segment folder to produce segment-level artifacts and metadata:
- Survey video (from `raw_images/`)
- Vertical displacement series (from `imu.txt`)
- Route length (from odometry/IMU log)
- Pothole depth timeline aligned to video (from pothole `inspection_result.json` files)
- GPS series (from `gps.txt`)
- Segment metadata (start/end location, length)

## Segment-folder structure (expected input)

Minimal required structure for one segment:

```
<segment_dir>/
  raw_images/
    <timestamp>.jpg
    <timestamp>.jpg
    ...
  gps.txt
  imu.txt
  <segment_scan>.pcd                   
  pothole_001/
    <timestamp>.jpg
    <timestamp>.pcd
    inspection_result.json
  pothole_002/
    ...
```

Notes:
- `raw_images/` is the reference timeline. The earliest image timestamp is t0.
- `gps.txt` format: `#timestamp latitude longitude [altitude]` per line.
- `imu.txt` format: `timestamp x y z ...` per line, where column 3 is vertical displacement.
- Pothole folders are discovered by name prefix (`pothole_`).
- For pothole depth, reads `inspection_result.json` (from `paired_assets_analysis`).

## Output files (written to segment_dir by default)

| File | Description |
|------|-------------|
| `output_video.mp4` | Survey video from `raw_images/` |
| `imu.json` | Vertical displacement series (sampled every 30s) |
| `route_length.json` | Route length in meters and kilometers |
| `segment_depth_timestamps.json` | Pothole depths aligned to video timeline |
| `segment_gps.json` | GPS entries as structured JSON |
| `segment_meta.json` | Aggregated metadata (start_loc, end_loc, length_in_km) |

## Domain model

### Entities
- `Segment`: Identity (`SegmentId`) + source directory path.

### Value Objects
- `SegmentId`: Unique identifier (folder name or UUID).
- `VideoArtifact`: path, fps, frame_count, duration_seconds.
- `VerticalDisplacementSeries`: list of `{video_timestamp, vertical_displacement}`.
- `RouteLengthResult`: meters, kilometers.
- `DepthTimeline`: list of `{video_timestamp, pothole_depth}`.
- `GpsSeries`: list of `{timestamp, lat, lng, alt}` with `start_location` / `end_location` properties.
- `SegmentMeta`: start_loc, end_loc, length_in_km.
- `SegmentProcessingResult`: aggregate of all artifacts + status + warnings/errors.
- `SegmentProcessingStatus`: OK, PARTIAL, FAILED.

## Ports (Protocol interfaces)

| Port | Responsibility |
|------|---------------|
| `VideoCreator` | Create video from images (side effect: writes mp4) |
| `ImuProcessor` | Process IMU log to vertical displacement series (side effect: writes JSON) |
| `RouteLengthCalculator` | Calculate route length from odometry (side effect: writes JSON) |
| `DepthTimelineExtractor` | Extract depth timeline from pothole inspections (side effect: writes JSON) |
| `GpsConverter` | Convert GPS text log to structured JSON (side effect: writes JSON) |
| `SegmentWorkspace` | Path resolution for inputs and outputs |
| `SegmentWorkspaceFactory` | Build workspace from segment directory |
| `SegmentArtifactsRepository` | Persist results (extensible for target tree export) |

All computation ports are side-effectful: they write files and return artifact receipts (path + data/metadata).

## Application layer

`ProcessSegmentUseCase` orchestrates the segment processing workflow.

### Typical flow

1. `ProcessSegmentUseCase.run(segment_dir, config)` is called.
2. `SegmentWorkspaceFactory.build(segment_dir)` resolves all input/output paths.
3. For each artifact:
   - `VideoCreator.create_video(...)` → `VideoArtifact`
   - `ImuProcessor.create_series(...)` → `VerticalDisplacementSeries`
   - `RouteLengthCalculator.calculate(...)` → `RouteLengthResult`
   - `DepthTimelineExtractor.extract(...)` → `DepthTimeline`
   - `GpsConverter.convert(...)` → `GpsSeries`
4. Build `SegmentMeta` from GPS + route length.
5. Construct `SegmentProcessingResult` with all artifacts + status.
6. Optionally persist via `SegmentArtifactsRepository.save(result)`.
7. Return result.

### Error handling

Each step is wrapped in try/except. Failures are logged and added to `result.errors`, but processing continues. Status is:
- `OK`: All critical artifacts produced.
- `PARTIAL`: Some artifacts missing but processing continued.
- `FAILED`: No artifacts produced.

## Infrastructure (Adapters)

| Adapter | Implements | Notes |
|---------|------------|-------|
| `OpenCVVideoCreator` | `VideoCreator` | Uses cv2 to create mp4 |
| `LegacyImuProcessor` | `ImuProcessor` | Wraps legacy `sensors.process_gps` logic |
| `LegacyRouteLengthCalculator` | `RouteLengthCalculator` | Wraps legacy `sensors.calculate_road_length` logic |
| `LegacyDepthTimelineExtractor` | `DepthTimelineExtractor` | Reads `inspection_result.json` |
| `FilesystemGpsConverter` | `GpsConverter` | Parses `gps.txt` |
| `FilesystemSegmentWorkspace` | `SegmentWorkspace` | Resolves filesystem paths |
| `FilesystemSegmentWorkspaceFactory` | `SegmentWorkspaceFactory` | Creates workspace instances |
| `FilesystemSegmentArtifactsRepository` | `SegmentArtifactsRepository` | No-op (ports persist); extensible |
| `Data2SegmentArtifactsRepository` | `SegmentArtifactsRepository` | Exports a legacy-compatible target tree under `output/data2` (writes/updates `copy_manifest.json` and `Roads/rd-*/rs-*`) |

## Usage

### CLI

```bash
python -m segment_processing.entrypoints.process_segment \
    --segment-dir /path/to/segment \
    --fps 10 \
    --imu-interval 30.0 \
    --log-level INFO
```

Export to legacy `output/data2` structure:

```bash
python -m segment_processing.entrypoints.process_segment \
    --segment-dir /path/to/segment \
    --fps 10 \
    --imu-interval 30.0 \
    --export-data2-root /path/to/indoScan/output/data2 \
    --log-level INFO
```

### Programmatic

```python
from pathlib import Path
from segment_processing.entrypoints.process_segment import process_segment

exit_code = process_segment(
    segment_dir=Path("/path/to/segment"),
    fps=10,
    imu_interval=30.0,
    persist=True,
)
```

### As a use case (with custom adapters)

```python
from segment_processing.application.use_case import (
    ProcessSegmentUseCase,
    ProcessSegmentConfig,
)
from segment_processing.infrastructure import (
    FilesystemSegmentWorkspaceFactory,
    OpenCVVideoCreator,
    LegacyImuProcessor,
    LegacyRouteLengthCalculator,
    LegacyDepthTimelineExtractor,
    FilesystemGpsConverter,
    Data2SegmentArtifactsRepository,
)

use_case = ProcessSegmentUseCase(
    workspace_factory=FilesystemSegmentWorkspaceFactory(),
    video_creator=OpenCVVideoCreator(),
    imu_processor=LegacyImuProcessor(),
    route_length_calculator=LegacyRouteLengthCalculator(),
    depth_timeline_extractor=LegacyDepthTimelineExtractor(),
    gps_converter=FilesystemGpsConverter(),
    artifacts_repository=Data2SegmentArtifactsRepository(export_root=Path("/path/to/indoScan/output/data2")),
)

result = use_case.run(
    segment_dir=Path("/path/to/segment"),
    config=ProcessSegmentConfig(fps=10, persist_results=False),
)

print(result.status)  # OK, PARTIAL, or FAILED
```

## Legacy compatibility

This context wraps and replaces:
- `run_segment_tasks.py` → `ProcessSegmentUseCase`
- `sensors/process_gps.py` → `LegacyImuProcessor`
- `sensors/calculate_road_length.py` → `LegacyRouteLengthCalculator`
- `sensors/segment_depth_timestamps.py` → `LegacyDepthTimelineExtractor`
- `media/create_video.py` → `OpenCVVideoCreator`

The legacy scripts can be deprecated once this context is validated.

## Future extensions

- `SegmentArtifactsRepository` has a legacy-compatible exporter (`Data2SegmentArtifactsRepository`) and can be further extended to:
  - Reuse stable IDs across different runs / machines (deterministic IDs instead of uuid4)
  - Upload to cloud storage
- `SegmentWorkspaceFactory` can support remote sources (S3, etc.) by materializing files locally before processing.
