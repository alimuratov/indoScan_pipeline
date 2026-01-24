from __future__ import annotations
from typing import Dict, Protocol
from assets_pairing.domain.model import ImageRef, PcdRef


class AssetSource(Protocol):
    """Port: enumerate available image/pcd assets keyed by stem."""

    def list_images(self) -> Dict[str, ImageRef]: ...
    def list_pcds(self) -> Dict[str, PcdRef]: ...


class SnapshotStore(Protocol):
    """Port: persist matched image/pcd pairs into a destination layout."""

    def prepare(self) -> None: ...
    def save_pair(self, *, pothole_id: int,
                  image: ImageRef, pcd: PcdRef) -> None: ...

    def finalize(self) -> None: ...
