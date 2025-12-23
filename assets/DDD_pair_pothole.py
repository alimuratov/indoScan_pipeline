from __future__ import annotations
from dataclasses import dataclass
from typing import Set, ClassVar, List, Tuple
from pathlib import Path
import shutil
from assets.repositories import KeyedRepository, SnapshotSink
import logging


@dataclass(frozen=True)
class PotholeSnapshot:
    id: int
    image: Image
    pcd: Pcd

    def __post_init__(self):
        # check if image and pcd are not empty
        if not self.image or not self.pcd:
            raise ValueError("Image and pcd must not be empty")
        if self.image.stem != self.pcd.stem:
            raise ValueError("Image and pcd must have the same stem")


class PotholeSnapshotRepository:
    def __init__(self, path: Path, zero_pad: int, move: bool):
        self._storage = []
        self.destination_directory_path = path
        self.zero_pad = zero_pad
        self.move = move

    def add(self, pothole_snapshot: PotholeSnapshot) -> None:
        self._storage.append(pothole_snapshot)

    def get(self, id: int) -> PotholeSnapshot:
        return self._storage[id]

    def save_all(self) -> None:
        # if the pothole folder already exists, delete it and its contents and create a new one
        for folder in self.destination_directory_path.iterdir():
            if folder.is_dir() and folder.name.startswith("pothole_"):
                shutil.rmtree(folder)

        for pothole_snapshot in self._storage:
            pothole_folder = self.destination_directory_path / \
                f"pothole_{pothole_snapshot.id:0{self.zero_pad}d}"

            pothole_folder.mkdir(parents=True, exist_ok=True)

            if self.move:
                shutil.move(str(pothole_snapshot.image.path),
                            pothole_folder / pothole_snapshot.image.path.name)
                shutil.move(str(pothole_snapshot.pcd.path),
                            pothole_folder / pothole_snapshot.pcd.path.name)
            else:
                shutil.copy2(str(pothole_snapshot.image.path),
                             pothole_folder / pothole_snapshot.image.path.name)
                shutil.copy2(str(pothole_snapshot.pcd.path),
                             pothole_folder / pothole_snapshot.pcd.path.name)


@dataclass(frozen=True)
class Pcd:
    pcd_exts: ClassVar[Set[str]] = {".pcd"}
    path: Path

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass(frozen=True)
class Image:
    image_exts: ClassVar[Set[str]] = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    path: Path

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def stem(self) -> str:
        return self.path.stem


class ImageRepository:
    def __init__(self):
        self._storage = {}

    def add(self, image: Image) -> None:
        if image.stem in self._storage:
            raise ValueError("Image with the same stem already exists")
        self._storage[image.stem] = image

    def get(self, stem: str) -> Image:
        return self._storage[stem]

    def keys(self) -> Set[str]:
        return set(self._storage.keys())


class PcdRepository:
    def __init__(self):
        self._storage = {}

    def add(self, pcd: Pcd) -> None:
        if pcd.stem in self._storage:
            raise ValueError("Pcd with the same stem already exists")
        self._storage[pcd.stem] = pcd

    def get(self, stem: str) -> Pcd:
        return self._storage[stem]

    def keys(self) -> Set[str]:
        return set(self._storage.keys())


class PairingService:
    def __init__(self, image_repository: KeyedRepository[Image], pcd_repository: KeyedRepository[Pcd], pothole_snapshot_repository: SnapshotSink[PotholeSnapshot], start_id: int):
        self._image_repository = image_repository
        self._pcd_repository = pcd_repository
        self._pothole_snapshot_repository = pothole_snapshot_repository
        self._start_id = start_id

    def pair(self, id: int, image: Image, pcd: Pcd) -> PotholeSnapshot:
        return PotholeSnapshot(id=id, image=image, pcd=pcd)

    def pair_all(self) -> Tuple[List[str], List[str]]:
        missing_image, missing_pcd = self.detect_missing_assets()

        matched = \
            self._image_repository.keys() & self._pcd_repository.keys()

        matched = sorted(matched)
        id = self._start_id
        for matched_key in matched:
            image = self._image_repository.get(matched_key)
            pcd = self._pcd_repository.get(matched_key)

            pothole_snapshot = self.pair(id, image, pcd)
            self._pothole_snapshot_repository.add(pothole_snapshot)

            id += 1

        return missing_image, missing_pcd

    def detect_missing_assets(self) -> Tuple[List[str], List[str]]:
        missing_pcd = self._image_repository.keys() - self._pcd_repository.keys()
        missing_image = self._pcd_repository.keys() - self._image_repository.keys()
        return sorted(missing_image), sorted(missing_pcd)


class ApplicationService:

    def run(self, image_dir: Path, pcd_dir: Path, output_dir: Path,
            start_id: int, zero_pad: int, move: bool, log_level: str) -> None:
        logging.basicConfig(
            level=getattr(logging, str(log_level).upper(), logging.INFO),
            format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        )

        # IO Layer
        image_repository = ImageRepository()
        pcd_repository = PcdRepository()

        for image_path in image_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in Image.image_exts:
                image = Image(path=image_path)
                image_repository.add(image)

        for pcd_path in pcd_dir.iterdir():
            if pcd_path.is_file() and pcd_path.suffix.lower() in Pcd.pcd_exts:
                pcd = Pcd(path=pcd_path)
                pcd_repository.add(pcd)

        pothole_snapshot_repository = PotholeSnapshotRepository(
            output_dir, zero_pad, move)

        # Logic Layer
        pairing_service = PairingService(
            image_repository, pcd_repository, pothole_snapshot_repository, start_id)
        missing_image, missing_pcd = pairing_service.pair_all()

        for missing_image in missing_image:
            logging.error("Missing image: %s",
                          missing_image)
        for missing_pcd in missing_pcd:
            logging.error("Missing pcd: %s",
                          missing_pcd)

        # IO Layer
        pothole_snapshot_repository.save_all()
