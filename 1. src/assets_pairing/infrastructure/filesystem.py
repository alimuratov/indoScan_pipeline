from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from assets_pairing.domain.model import ImageRef, PcdRef


@dataclass(frozen=True)
class FilesystemAssetSource:
    """Filesystem adapter: enumerate assets from two flat directories."""

    image_dir: Path
    pcd_dir: Path

    def list_images(self) -> Dict[str, ImageRef]:
        out: Dict[str, ImageRef] = {}
        for p in self.image_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in ImageRef.image_exts:
                continue
            ref = ImageRef(path=p)
            out[ref.stem] = ref
        return out

    def list_pcds(self) -> Dict[str, PcdRef]:
        out: Dict[str, PcdRef] = {}
        for p in self.pcd_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in PcdRef.pcd_exts:
                continue
            ref = PcdRef(path=p)
            out[ref.stem] = ref
        return out


@dataclass
class FilesystemSnapshotStore:
    """Filesystem adapter: persist pairs into `pothole_<id>` folders."""

    destination_directory_path: Path
    zero_pad: int = 3
    move: bool = False

    def prepare(self) -> None:
        # Create destination dir (or clean old pothole_* subfolders)
        if not self.destination_directory_path.is_dir():
            self.destination_directory_path.mkdir(parents=True, exist_ok=True)
            return

        for folder in self.destination_directory_path.iterdir():
            if folder.is_dir() and folder.name.startswith("pothole_"):
                shutil.rmtree(folder)

    def save_pair(self, *, pothole_id: int, image: ImageRef, pcd: PcdRef) -> None:
        pothole_folder = self.destination_directory_path / \
            f"pothole_{pothole_id:0{self.zero_pad}d}"
        pothole_folder.mkdir(parents=True, exist_ok=True)

        img_dst = pothole_folder / image.path.name
        pcd_dst = pothole_folder / pcd.path.name

        if self.move:
            shutil.move(str(image.path), img_dst)
            shutil.move(str(pcd.path), pcd_dst)
        else:
            shutil.copy2(str(image.path), img_dst)
            shutil.copy2(str(pcd.path), pcd_dst)

    def finalize(self) -> None:
        # No-op for filesystem implementation (kept for symmetry/testing).
        return
