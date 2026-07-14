from collections.abc import Iterable, Sequence

from .inference import HistoryItem


class StepHistory:
    """A run's conversation history: the list of ``ToolCallDecision`` /
    ``ToolCallResult`` items rendered back into the prompt each step.

    Pure in-memory state — loading, persistence, and the step count live on the
    executor. The executor appends completed steps via :meth:`extend`; step
    middleware may reshape the history (compaction) via :meth:`rewrite`.
    ``items`` returns a cached immutable tuple, so reads are O(1).

    :attr:`dirty` flags a mid-step :meth:`rewrite` so the executor can persist
    the reshaped history before the model call — so a resume loads it instead of
    re-running the (possibly expensive) compactor.
    """

    def __init__(self, items: Sequence[HistoryItem] = ()):
        self._items = list(items)
        self._snapshot = tuple(self._items)
        self._dirty = False

    @property
    def items(self) -> tuple[HistoryItem, ...]:
        return self._snapshot

    @property
    def dirty(self) -> bool:
        """Whether :meth:`rewrite` ran since the last :meth:`mark_persisted`."""
        return self._dirty

    def mark_persisted(self) -> None:
        self._dirty = False

    def extend(self, items: Iterable[HistoryItem]) -> None:
        self._items.extend(items)
        self._snapshot = tuple(self._items)

    def rewrite(self, items: Sequence[HistoryItem]) -> None:
        """Replace the history (e.g. compaction). Does not advance the run."""
        self._items = list(items)
        self._snapshot = tuple(self._items)
        self._dirty = True
