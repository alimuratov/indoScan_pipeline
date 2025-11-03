from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .fs import list_immediate_subdirs

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def is_pothole_dir(path: str) -> bool:
    path = Path(path).resolve()
    try:
        names = {p.name.lower() for p in path.iterdir() if p.is_file()}
    except Exception:
        return False
    has_pcd = any(n.endswith(".pcd") for n in names)
    has_img = any(any(n.endswith(ext) for ext in IMAGE_EXTS) for n in names)
    return has_pcd and has_img

def discover_pothole_dirs(root: Path) -> List[Path]:
    if is_pothole_dir(root):
        return [root]
    out: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        d = Path(dirpath)
        if is_pothole_dir(d):
            out.append(d)
    return out


def is_segment_dir(path: str) -> bool:
    try:
        for child in list_immediate_subdirs(path):
            if is_pothole_dir(child):
                return True
    except Exception:
        return False
    return False


def is_road_dir(path: str) -> bool:
    try:
        for child in list_immediate_subdirs(path):
            if is_segment_dir(child):
                return True
    except Exception:
        return False
    return False


def discover_roads(roads_root: str) -> List[str]:
    roads: List[str] = []
    for maybe_road in list_immediate_subdirs(roads_root):
        if is_road_dir(maybe_road):
            roads.append(maybe_road)
    return roads

def discover_target_roads(target_roads_dir: str) -> List[str]:
    """Discover road directories under the target tree (IDs are folder names)."""
    try:
        return [
            os.path.join(target_roads_dir, name)
            for name in os.listdir(target_roads_dir)
            if os.path.isdir(os.path.join(target_roads_dir, name))
        ]
    except Exception:
        return []


def discover_target_segments(road_dir: str) -> List[str]:
    try:
        return [
            os.path.join(road_dir, name)
            for name in os.listdir(road_dir)
            if os.path.isdir(os.path.join(road_dir, name)) and is_target_segment_dir(os.path.join(road_dir, name))
        ]
    except Exception:
        return []

def discover_segments(root: str) -> List[str]:
    segments: List[str] = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if is_segment_dir(os.path.join(dirpath, d)):
                segments.append(os.path.join(dirpath, d))
    return segments

