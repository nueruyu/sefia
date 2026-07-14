from collections.abc import Iterable, Sequence

from .inference import HistoryItem


class StepHistory:
    """A run's conversation history: the list of ``ToolCallDecision`` /
    ``ToolCallResult`` items rendered back into the prompt each step.

    Pure in-memory state — loading, persistence, and the step count live on the
    executor. The executor appends completed steps via :meth:`extend`; step
    middleware may reshape the history (compaction) via :meth:`rewrite`.
    ``items`` returns a cached immutable tuple, so reads are O(1).
    """

    def __init__(self, items: Sequence[HistoryItem] = ()):
        self._items = list(items)
        self._snapshot = tuple(self._items)

    @property
    def items(self) -> tuple[HistoryItem, ...]:
        return self._snapshot

    def extend(self, items: Iterable[HistoryItem]) -> None:
        self._items.extend(items)
        self._snapshot = tuple(self._items)

    def rewrite(self, items: Sequence[HistoryItem]) -> None:
        """Replace the history (e.g. compaction). Does not advance the run."""
        self._items = list(items)
        self._snapshot = tuple(self._items)
