from collections.abc import Sequence

from ._interfaces.history_store import HistoryStore
from .inference import HistoryItem


class TransientHistoryStore(HistoryStore):
    """A no-op store that leaves history derived from replay."""

    async def load(self) -> list[HistoryItem]:
        return []

    async def save(self, items: Sequence[HistoryItem]) -> None:
        pass
