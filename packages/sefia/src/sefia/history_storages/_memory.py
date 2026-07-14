from .._interfaces.history_storage import HistorySnapshot, HistoryStorage


class MemoryHistoryStorage(HistoryStorage):
    """An in-memory :class:`HistoryStorage` for tests and embedding; keeps every
    saved snapshot in :attr:`saves`."""

    def __init__(self, initial: HistorySnapshot | None = None):
        self.snapshot = initial if initial is not None else HistorySnapshot()
        self.saves: list[HistorySnapshot] = []

    async def load(self) -> HistorySnapshot:
        return self.snapshot

    async def save(self, snapshot: HistorySnapshot) -> None:
        self.snapshot = snapshot
        self.saves.append(snapshot)
