from collections.abc import Iterable, Sequence

from .inference import HistoryItem


class StepHistory:
    """A run's conversation history, rendered into the prompt each step.

    Pure in-memory state; the executor owns loading, persistence, and the step
    count. It appends steps via :meth:`extend`; middleware reshapes the history
    via :meth:`rewrite`. Both set :attr:`dirty`, which the executor consults to
    persist unsaved changes. ``items`` is a cached immutable tuple.
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
        return self._dirty

    def mark_persisted(self) -> None:
        self._dirty = False

    def extend(self, items: Iterable[HistoryItem]) -> None:
        self._items.extend(items)
        self._snapshot = tuple(self._items)
        self._dirty = True

    def rewrite(self, items: Sequence[HistoryItem]) -> None:
        self._items = list(items)
        self._snapshot = tuple(self._items)
        self._dirty = True
