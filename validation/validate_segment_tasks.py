from typing import List
import os
from validation.validation_helpers import ValidationIssue
from common.discovery import discover_segments, is_pothole_dir
from common.config import Config

def validate_workspace_root_exists(workspace_root: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.exists(workspace_root):
        issues.append(ValidationIssue(workspace_root, "WORKSPACE_ROOT_MISSING", "error", "Workspace root does not exist"))
    return issues

def validate_source_roads_root_exists(workspace_root: str, source_roads_root: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.exists(os.path.join(workspace_root, source_roads_root)):
        issues.append(ValidationIssue(os.path.join(workspace_root, source_roads_root), "SOURCE_ROADS_ROOT_MISSING", "error", "Source roads root does not exist"))
    return issues

def validate_target_roads_root(workspace_root: str, target_roads_root: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.exists(os.path.join(workspace_root, target_roads_root)):
        issues.append(ValidationIssue(target_roads_root, "TARGET_ROADS_ROOT_MISSING", "error", "Target roads root does not exist"))
    return issues


def validate_scripts_root_exists(workspace_root: str, scripts_root: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.exists(os.path.join(workspace_root, scripts_root)):
        issues.append(ValidationIssue(scripts_root, "SCRIPTS_ROOT_MISSING", "error", "Scripts root does not exist"))
    return issues

def validate_estimate_script_exists(workspace_root: str, scripts_root: str, estimate_script: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.exists(os.path.join(workspace_root, scripts_root, estimate_script)):
        issues.append(ValidationIssue(estimate_script, "ESTIMATE_SCRIPT_MISSING", "error", "Estimate script does not exist"))
    return issues

def validate_gps_file_exists(segment_dir: str, gps_filename: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.exists(os.path.join(segment_dir, gps_filename)):
        issues.append(ValidationIssue(os.path.join(segment_dir, gps_filename), "GPS_FILE_MISSING", "error", "GPS file does not exist"))
    return issues

def validate_imu_file_exists(segment_dir: str, imu_filename: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.exists(os.path.join(segment_dir, imu_filename)):
        issues.append(ValidationIssue(os.path.join(segment_dir, imu_filename), "IMU_FILE_MISSING", "error", "IMU file does not exist"))
    return issues

def validate_raw_images_folder_exists(segment_dir: str, raw_images_folder_name: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.exists(os.path.join(segment_dir, raw_images_folder_name)):
        issues.append(ValidationIssue(os.path.join(segment_dir, raw_images_folder_name), "RAW_IMAGES_MISSING", "error", "Raw images folder does not exist"))
    return issues

def validate_segment_pcd_exists(segment_dir: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for file in os.listdir(segment_dir):
        if file.endswith(".pcd"):
            return issues
    issues.append(ValidationIssue(segment_dir, "PCD_MISSING", "error", "No pcd files found in segment"))
    return issues


def validate_source_roads_root_before_processing(config: Config) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    issues.extend(validate_workspace_root_exists(config.paths.workspace_root)) 
    issues.extend(validate_source_roads_root_exists(config.paths.workspace_root, config.paths.source_roads_root)) 

    for segment_dir in discover_segments(os.path.join(config.paths.workspace_root, config.paths.source_roads_root)):
        issues.extend(validate_gps_file_exists(segment_dir, config.paths.gps_filename))
        issues.extend(validate_imu_file_exists(segment_dir, config.paths.imu_filename))
        issues.extend(validate_raw_images_folder_exists(segment_dir, config.paths.images_folder_name))
        issues.extend(validate_segment_pcd_exists(segment_dir))
    return issues

def validate_segment_depth_timestamps_exists(segment_dir: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.exists(os.path.join(segment_dir, "segment_depth_timestamps.json")):
        issues.append(ValidationIssue(os.path.join(segment_dir, "segment_depth_timestamps.json"), "SEGMENT_DEPTH_TIMESTAMPS_MISSING", "error", "Segment depth timestamps does not exist"))
    return issues

def validate_imu_json_exists(segment_dir: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.exists(os.path.join(segment_dir, "imu.json")):
        issues.append(ValidationIssue(os.path.join(segment_dir, "imu.json"), "IMU_JSON_MISSING", "error", "IMU json does not exist"))
    return issues

def validate_route_length_exists(segment_dir: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.exists(os.path.join(segment_dir, "route_length.json")):
        issues.append(ValidationIssue(os.path.join(segment_dir, "route_length.json"), "ROUTE_LENGTH_MISSING", "error", "Route length does not exist"))
    return issues

def validate_output_video_exists(segment_dir: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not os.path.exists(os.path.join(segment_dir, "output_video.mp4")):
        issues.append(ValidationIssue(os.path.join(segment_dir, "output_video.mp4"), "OUTPUT_VIDEO_MISSING", "error", "Output video does not exist"))
    return issues

def validate_output_txt_exists(segment_dir: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for pothole_dir in os.listdir(segment_dir):
        if is_pothole_dir(pothole_dir):
            if not os.path.exists(os.path.join(segment_dir, pothole_dir, "output.txt")):
                issues.append(ValidationIssue(os.path.join(segment_dir, pothole_dir, "output.txt"), "OUTPUT_TXT_MISSING", "error", "Output txt does not exist"))
    return issues

def validate_source_roads_root_after_processing(config: Config) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    for segment_dir in discover_segments(os.path.join(config.paths.workspace_root, config.paths.source_roads_root)):
        issues.extend(validate_segment_depth_timestamps_exists(segment_dir))
        issues.extend(validate_imu_json_exists(segment_dir))
        issues.extend(validate_route_length_exists(segment_dir))
        issues.extend(validate_output_video_exists(segment_dir))
    return issues