from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..inference import HistoryItem


@dataclass(frozen=True)
class HistorySnapshot:
    """Persisted history and its compaction-independent progress counter."""

    items: tuple[HistoryItem, ...] = ()
    completed_steps: int = 0


class HistoryStorage(ABC):
    """Persistence backend for an inference run's history snapshots."""

    @abstractmethod
    async def load(self) -> HistorySnapshot:
        """Return the stored snapshot for the current run (empty if none)."""
        ...

    @abstractmethod
    async def save(self, snapshot: HistorySnapshot) -> None:
        """Persist ``snapshot`` as the current run's complete history."""
        ...
