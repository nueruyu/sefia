from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..inference import HistoryItem


@dataclass(frozen=True)
class HistorySnapshot:
    """A run's persisted conversation history plus its progress counter.

    ``items`` is the accumulated ``ToolCallDecision`` / ``ToolCallResult`` list
    rendered back into the prompt each step. ``completed_steps`` is the number
    of steps already finished — the run's authoritative progress, tracked
    separately from ``len(items)`` so that compaction (which shrinks ``items``)
    does not distort the step count on resume.
    """

    items: tuple[HistoryItem, ...] = ()
    completed_steps: int = 0


class HistoryStorage(ABC):
    """
    The persistence backend for an inference run's history — the *raw* seam,
    analogous to sefios' ``SessionStorage`` (whereas :class:`HistoryStore` is
    the higher-level service built on top of it).

    Both methods run inside the engraved ``@infer`` run, so an implementation
    can identify the current run via ``glyff.get_context().current_execution_id``
    (see :class:`~sefia.GlyffHistoryStorage`). ``save`` receives a full snapshot
    each time; writes must commit immediately so a snapshot taken before a pause
    survives the interrupt. The default :class:`~sefia.GlyffHistoryStorage`
    stores the snapshot in the run execution's glyff metadata, so history is
    durable and a resumed run continues from the saved step without replaying
    the completed ones.
    """

    @abstractmethod
    async def load(self) -> HistorySnapshot:
        """Return the stored snapshot for the current run (empty if none)."""
        ...

    @abstractmethod
    async def save(self, snapshot: HistorySnapshot) -> None:
        """Persist ``snapshot`` as the current run's complete history."""
        ...
