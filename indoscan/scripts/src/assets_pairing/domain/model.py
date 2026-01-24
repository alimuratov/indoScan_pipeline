from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Set


@dataclass(frozen=True)
class ImageRef:
    """Value object referencing an image file on disk."""

    image_exts: ClassVar[Set[str]] = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    }

    path: Path

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass(frozen=True)
class PcdRef:
    """Value object referencing a point cloud file on disk."""

    pcd_exts: ClassVar[Set[str]] = {".pcd"}

    path: Path

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass(frozen=True)
class PairingResult:
    """Value object describing the outcome of pairing by stem."""

    matched_keys: List[str]
    missing_images: List[str]  # stems present in PCDs but not images
    missing_pcds: List[str]  # stems present in images but not PCDs
