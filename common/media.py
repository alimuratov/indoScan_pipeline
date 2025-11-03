from __future__ import annotations
from typing import Dict, Tuple
from common.fs import copy_asset

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

def copy_segment_media(segment_dir: str, target_segment_dir: str) -> Dict[str, str]:
    results: Dict[str, str] = {}
    survey_src = detect_segment_survey_video(segment_dir)
    if survey_src:
        dst = os.path.join(target_segment_dir, "Data", "Survey video", os.path.basename(survey_src))
        if copy_asset(survey_src, dst):
            results["survey_video"] = dst
    lidar_src = detect_segment_lidar_scan(segment_dir)
    if lidar_src:
        dst = os.path.join(target_segment_dir, "Data", "Lidar Scan", os.path.basename(lidar_src))
        if copy_asset(lidar_src, dst):
            results["lidar_scan"] = dst
    return results

def find_pothole_media(pothole_dir: str) -> Tuple[str, str]:
    img_src = None
    pcd_src = None
    for fname in os.listdir(pothole_dir):
        if fname.endswith(".jpg") or fname.endswith(".png"):
            img_src = os.path.join(pothole_dir, fname)
        elif fname.endswith(".pcd"):
            pcd_src = os.path.join(pothole_dir, fname)
    if img_src is None or pcd_src is None:
        raise ValueError(f"No media found in {pothole_dir}")
    return img_src, pcd_src