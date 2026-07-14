"""Shared test doubles for the sefia unit tests."""

from sefia import HistorySnapshot, HistoryStorage


class MemoryHistoryStorage(HistoryStorage):
    """In-memory `HistoryStorage`; records every saved snapshot in `saves`."""

    def __init__(self, initial: HistorySnapshot | None = None):
        self.snapshot = initial if initial is not None else HistorySnapshot()
        self.saves: list[HistorySnapshot] = []

    async def load(self) -> HistorySnapshot:
        return self.snapshot

    async def save(self, snapshot: HistorySnapshot) -> None:
        self.snapshot = snapshot
        self.saves.append(snapshot)
