from collections.abc import Sequence

from ._interfaces.history_storage import HistorySnapshot, HistoryStorage
from .inference import HistoryItem


class StepHistory:
    """The executor's mutable run history, backed by a :class:`HistoryStorage`.

    Middleware sees only the read-plus-``rewrite`` surface (via the
    :class:`~sefia.History` protocol); the executor additionally drives
    :meth:`load` and :meth:`record_step`.
    """

    def __init__(self, storage: HistoryStorage):
        self._storage = storage
        self._items: list[HistoryItem] = []
        self._completed_steps = 0

    @property
    def items(self) -> Sequence[HistoryItem]:
        return tuple(self._items)

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    async def rewrite(self, items: Sequence[HistoryItem]) -> None:
        """Persist and replace history without advancing the step count."""
        new_items = list(items)
        await self._storage.save(
            HistorySnapshot(tuple(new_items), self._completed_steps)
        )
        self._items[:] = new_items

    async def load(self) -> None:
        snapshot = await self._storage.load()
        self._items = list(snapshot.items)
        self._completed_steps = snapshot.completed_steps

    async def record_step(
        self, decision: HistoryItem, results: Sequence[HistoryItem]
    ) -> None:
        self._items.append(decision)
        self._items.extend(results)
        self._completed_steps += 1
        await self._storage.save(
            HistorySnapshot(tuple(self._items), self._completed_steps)
        )
