"""Tests for assets_pairing domain services."""

import pytest
from pathlib import Path

from assets_pairing.domain.model import ImageRef, PcdRef, PairingResult
from assets_pairing.domain.services import PairingService


class FakeSnapshotStore:
    """Test double for SnapshotStore port."""

    def __init__(self):
        self.saved_pairs = []

    def prepare(self) -> None:
        pass

    def save_pair(self, *, pothole_id: int, image: ImageRef, pcd: PcdRef) -> None:
        self.saved_pairs.append({
            "id": pothole_id,
            "image": image,
            "pcd": pcd,
        })

    def finalize(self) -> None:
        pass


def test_pairing_service_match_correct_assets():
    """PairingService matches assets by stem and reports missing ones."""
    image_stems = {"file1", "file2"}
    pcd_stems = {"file1", "file3"}

    result = PairingService().pair_assets(image_stems, pcd_stems)

    assert result.matched_keys == ["file1"]
    assert result.missing_images == ["file3"]  # PCDs without matching images
    assert result.missing_pcds == ["file2"]  # images without matching PCDs


def test_pairing_service_pairs_multiple_matches_sorted():
    """PairingService returns matched keys sorted alphabetically."""
    image_stems = {"b", "a", "c"}
    pcd_stems = {"b", "a", "c"}

    result = PairingService().pair_assets(image_stems, pcd_stems)

    assert result.matched_keys == ["a", "b", "c"]
    assert result.missing_images == []
    assert result.missing_pcds == []


def test_pairing_service_returns_missing_lists_sorted():
    """PairingService returns missing lists sorted alphabetically."""
    image_stems = {"a", "b"}
    pcd_stems = {"b", "c"}

    result = PairingService().pair_assets(image_stems, pcd_stems)

    assert result.matched_keys == ["b"]
    assert result.missing_images == ["c"]
    assert result.missing_pcds == ["a"]


def test_pairing_result_is_frozen():
    """PairingResult is immutable."""
    result = PairingResult(
        matched_keys=["a"],
        missing_images=["b"],
        missing_pcds=["c"],
    )
    with pytest.raises(AttributeError):
        result.matched_keys = ["x"]


def test_image_ref_and_pcd_ref_properties():
    """ImageRef and PcdRef expose stem and extension properties."""
    img = ImageRef(path=Path("some/dir/file1.jpg"))
    pcd = PcdRef(path=Path("some/dir/file1.pcd"))

    assert img.stem == "file1"
    assert img.extension == ".jpg"
    assert pcd.stem == "file1"
    assert pcd.extension == ".pcd"
