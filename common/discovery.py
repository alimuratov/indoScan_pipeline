from __future__ import annotations

import os
from typing import List

from .fs import list_immediate_subdirs


def is_pothole_dir(path: str) -> bool:
    try:
        names = set(os.listdir(path))
    except Exception:
        return False
    has_artifact = any(
        any(name.lower().endswith(ext) for ext in (".jpg", ".png", ".pcd"))
        for name in names
    ) or ("output.txt" in names)
    return has_artifact


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