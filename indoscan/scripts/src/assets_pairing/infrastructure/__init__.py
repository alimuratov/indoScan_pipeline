"""Infrastructure layer: concrete IO implementations for assets pairing."""

from .filesystem import FilesystemAssetSource, FilesystemSnapshotStore

__all__ = ["FilesystemAssetSource", "FilesystemSnapshotStore"]
