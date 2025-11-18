1. Run `run_pre_processing.py` to pair pothole assets under `pothole_<id>`folders in the segment directories

$ indoscan: python scripts/orchestration/run_pre_processing.py --config ./config/config.yaml

Relevant config parameters:
- paths.workspace_root: Path to the workspace root
- paths.source_roads_root: Name of the source roads root
- pre_processing.images_dir: Name of the images directory
- pre_processing.pcd_dir: Name of the pcd directory

Source roads root structure:

```
source_roads_root/
├── segment_1/
│   ├── Images/
│   ├── PCDs/
├── segment_2/
├── ...
```

Images and PCDs directories must contain images and point clouds respectively. Images and point clouds must be named using the corresponding timestamp (e.g., 1756369693.599339008.jpg and 1756369693.599339008.pcd). 

Expected output:

```
source_roads_root/
├── segment_1/
│   ├── Images/
│   ├── PCDs/
│   ├── pothole_<id>/
│   │   ├── <timestamp>.jpg
│   │   ├── <timestamp>.pcd
│   ├── pothole_<id>/
│   ├── ...
├── segment_2/
├── ...
```

Validation scripts will check for the following issues:
- Duplicate images or PCDs
- Missing images or PCDs
- Mismatched images and PCDs

2. Manually transfer gps.txt, imu.txt, segment point cloud and raw_images folder to the segment direcotries
3. Run `run_segment_tasks.py` to process the segment-level tasks (calculate route length, create video, etc.)
This will create 
- `segment_depth_timestamps.json`,
- `vt_gps_z.json` (Vertical displacements from GPS, timed according to the video timeline.),
- `route_length.json` and
- `output_video.mp4` in each segment directory

$ indoscan python scripts/orchestration/run_segment_tasks.py --config ./config/config.yaml

Relevant config parameters:
- paths.source_roads_root: Path to the source roads root
- paths.images_folder_name: Name of the images directory
- paths.odometry_filename: Name of the odometry file
- media.fps: Frames per second for the video
- logging.level: Logging level

Source roads root structure:

```
source_roads_root/
├── segment_1/
│   ├── Images/
│   ├── PCDs/
│   ├── pothole_<id>/
│   │   ├── <timestamp>.jpg
│   │   ├── <timestamp>.pcd
│   ├── pothole_<id>/
│   ├── <gps_filename>.txt
│   ├── <imu_filename>.txt
├── segment_2/
├── ...
```

Expected output:

```
source_roads_root/
├── segment_1/
│   ├── Images/
│   ├── PCDs/
│   ├── pothole_<id>/
│   │   ├── <timestamp>.jpg
│   │   ├── <timestamp>.pcd
│   │   ├── output.txt
│   ├── pothole_<id>/
│   ├── ...
│   ├──  aggregated_depth_data.json
│   ├──  imu.json
│   ├──  route_length.json
│   ├──  output_video.mp4
├── segment_2/
├── ...
```

Validation scripts will check for the following issues:
- Missing segment depth timestamps data
- Missing imu data
- Missing route length data
- Missing video
- Missing output.txt in the pothole directories
