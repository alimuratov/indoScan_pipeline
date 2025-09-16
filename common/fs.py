from __future__ import annotations

import os
from typing import List, Tuple, Optional
import shutil
import logging
import shutil
import logging


def list_immediate_subdirs(parent_dir: str) -> List[str]:
    return [
        os.path.join(parent_dir, name)
        for name in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, name))
    ]


def find_files_with_extensions(root_dir: str, exts: Tuple[str, ...]) -> List[str]:
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(exts):
                matches.append(os.path.join(dirpath, fname))
    return matches


def find_immediate_files_with_extensions(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    matches: List[str] = []
    try:
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(exts):
                matches.append(fpath)
    except Exception:
        pass
    return matches


def is_dir(path: str) -> bool:
    try:
        return os.path.isdir(path)
    except Exception:
        return False


def ensure_parent_dir(path: str) -> None:
    """Ensure parent directory exists for a given file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def copy_asset(src_path: str, dst_abs_path: str) -> bool:
    """Copy a file to destination, logging success/failure."""
    try:
        ensure_parent_dir(dst_abs_path)
        shutil.copy2(src_path, dst_abs_path)
        logging.info("Copied asset: %s -> %s", src_path, dst_abs_path)
        return True
    except Exception:
        logging.exception("Failed to copy asset: %s -> %s", src_path, dst_abs_path)
        return False


def find_first_matching_file(root_dir: str, patterns: Tuple[str, ...]) -> Optional[str]:
    """Find first file whose lowercase name contains any pattern."""
    patterns_lower = tuple(p.lower() for p in patterns)
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            low = fname.lower()
            if any(p in low for p in patterns_lower):
                return os.path.join(dirpath, fname)
    return None

def to_relpath(path: str, relative_to_dir: str) -> str:
    """Return path relative to a directory, preserving path separators.

    Interaction: Central helper to produce JSON paths relative to
    --target-roads-dir after assets are copied into the target tree.
    """
    try:
        return os.path.relpath(path, start=relative_to_dir)
    except Exception:
        logging.debug("to_relpath fallback for path=%s relative_to=%s", path, relative_to_dir)
        return path


def join_target(*parts: str) -> str:
    """Join path parts for target tree layout.

    Interaction: Used by segment/pothole builders to calculate destination
    asset paths (videos, lidar, images) under the target Roads directory.
    """
    return os.path.join(*parts)

def ensure_parent_dir(path: str) -> None:
    """Ensure parent directory exists for a given file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def copy_asset(src_path: str, dst_abs_path: str) -> bool:
    """Copy a file to destination, logging success/failure.

    Interaction: Called when copying survey videos, lidar scans, and images
    into the target tree so that JSON can reference them consistently.
    """
    try:
        ensure_parent_dir(dst_abs_path)
        shutil.copy2(src_path, dst_abs_path)
        logging.info("Copied asset: %s -> %s", src_path, dst_abs_path)
        return True
    except Exception:
        logging.exception("Failed to copy asset: %s -> %s", src_path, dst_abs_path)
        return False

def find_first_matching_file(root_dir: str, patterns: Tuple[str, ...]) -> Optional[str]:
    """Find first file whose lowercase name contains any pattern.

    Note: Not currently used by the main flow but kept for potential extension.
    """
    patterns_lower = tuple(p.lower() for p in patterns)
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            low = fname.lower()
            if any(p in low for p in patterns_lower):
                return os.path.join(dirpath, fname)
    return None
