from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..inference import HistoryItem


class HistoryStore(ABC):
    """
    The source of truth for an inference run's conversation history.

    By default history is *derived* state: the executor rebuilds it by
    replaying the run's engraved steps, whose durable records are keyed on the
    history contents. A ``HistoryStore`` makes history *stored* state instead:
    the executor loads it at the start of each attempt and saves a snapshot
    after every completed step, so a resumed invocation continues from the
    persisted history without replaying every prior step, and the history can
    be rewritten (compacted) independently of glyff's execution log.

    Both methods are called inside the engraved ``@infer`` run, so an
    implementation can identify the run via
    ``glyff.get_context().current_execution_id`` (see sefios'
    ``DurableHistoryStore``). ``save`` receives the full history each time —
    a full snapshot is trivially idempotent, which keeps the crash contract
    simple: the executor persists a step's decision and tool results only
    after the step's engraved calls committed, so a resume that re-loads an
    older snapshot replays the missing step from its engraved record and saves
    again.

    Note that with a persistent store, a retry attempt (an inference
    middleware calling the run again) continues from the last saved history
    instead of starting the attempt from scratch.
    """

    @abstractmethod
    async def load(self) -> list[HistoryItem]:
        """Return the stored history for the current run (empty if none)."""
        ...

    @abstractmethod
    async def save(self, items: Sequence[HistoryItem]) -> None:
        """Persist ``items`` as the current run's complete history."""
        ...
