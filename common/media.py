from __future__ import annotations

import os
from .fs import find_immediate_files_with_extensions, find_files_with_extensions


def detect_segment_survey_video(segment_dir: str) -> str:
    video_exts = (".mp4", ".mov", ".mkv", ".avi")
    immediate = find_immediate_files_with_extensions(segment_dir, video_exts)
    if immediate:
        return immediate[0]
    subfolders = [
        os.path.join(segment_dir, "Data/Survey video"),
        os.path.join(segment_dir, "Data/survey_video"),
        os.path.join(segment_dir, "Survey video"),
    ]
    for base in subfolders:
        if os.path.isdir(base):
            vids = find_files_with_extensions(base, video_exts)
            if vids:
                return vids[0]
    return ""


def detect_segment_lidar_scan(segment_dir: str) -> str:
    immediate_pcds = find_immediate_files_with_extensions(segment_dir, (".pcd",))
    if immediate_pcds:
        return immediate_pcds[0]
    subfolders = [
        os.path.join(segment_dir, "Data/Lidar Scan"),
        os.path.join(segment_dir, "Data/lidar_scan"),
        os.path.join(segment_dir, "Lidar Scan"),
    ]
    for base in subfolders:
        if os.path.isdir(base):
            pcds = find_files_with_extensions(base, (".pcd",))
            if pcds:
                return pcds[0]
    return ""


