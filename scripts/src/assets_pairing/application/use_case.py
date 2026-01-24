from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from assets_pairing.domain.model import PairingResult
from assets_pairing.domain.services import PairingService
from assets_pairing.ports import AssetSource, SnapshotStore


@dataclass(frozen=True)
class PairAssetsSummary:
    pairing: PairingResult
    written_pairs: int


class PairAssetsUseCase:
    """Use case: pair images with PCDs by stem and persist them as pothole folders."""

    def __init__(
        self,
        *,
        asset_source: AssetSource,
        snapshot_store: SnapshotStore,
        pairing_service: Optional[PairingService] = None,
    ) -> None:
        self._asset_source = asset_source
        self._snapshot_store = snapshot_store
        self._pairing_service = pairing_service or PairingService()

    def run(self, *, start_id: int = 1) -> PairAssetsSummary:
        images = self._asset_source.list_images()
        pcds = self._asset_source.list_pcds()

        pairing = self._pairing_service.pair_assets(
            set(images.keys()), set(pcds.keys()))

        self._snapshot_store.prepare()

        pothole_id = start_id
        written = 0
        for stem in pairing.matched_keys:
            self._snapshot_store.save_pair(
                pothole_id=pothole_id,
                image=images[stem],
                pcd=pcds[stem],
            )
            written += 1
            pothole_id += 1

        self._snapshot_store.finalize()
        return PairAssetsSummary(pairing=pairing, written_pairs=written)
