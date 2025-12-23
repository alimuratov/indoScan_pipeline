from typing import Protocol, TypeVar, Generic, Set

# a placeholder type
T = TypeVar('T')


class KeyedRepository(Protocol[T]):
    def get(self, key: str) -> T: ...
    def keys(self) -> Set[str]: ...


class SnapshotSink(Protocol[T]):
    def add(self, snapshot: T) -> None: ...


class SnapshotStore(SnapshotSink[T], Protocol[T]):
    def save_all(self) -> None: ...
