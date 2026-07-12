from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..inference import HistoryItem


class HistoryStore(ABC):
    """
    Stores complete conversation-history snapshots for one inference run.

    Methods run inside the engraved ``@infer`` call, allowing implementations
    to scope data with ``glyff.get_context().current_execution_id``.
    """

    @abstractmethod
    async def load(self) -> list[HistoryItem]:
        """Return the stored history for the current run (empty if none)."""
        ...

    @abstractmethod
    async def save(self, items: Sequence[HistoryItem]) -> None:
        """Persist ``items`` as the current run's complete history."""
        ...
