from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from .model import PairingResult


@dataclass(frozen=True)
class PairingService:
    """Domain service: pure pairing logic by filename stem."""

    def pair_assets(self, image_stems: Set[str], pcd_stems: Set[str]) -> PairingResult:
        matched = sorted(image_stems & pcd_stems)
        missing_images = sorted(pcd_stems - image_stems)
        missing_pcds = sorted(image_stems - pcd_stems)
        return PairingResult(
            matched_keys=matched,
            missing_images=missing_images,
            missing_pcds=missing_pcds,
        )
