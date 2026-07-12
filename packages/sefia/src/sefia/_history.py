from collections.abc import Sequence

import glyff

from ._interfaces.history_storage import HistorySnapshot, HistoryStorage
from .inference import HistoryItem

_METADATA_KEY = "sefia:history"


class GlyffHistoryStorage(HistoryStorage):
    """Stores history in the current run's glyff metadata."""

    async def load(self) -> HistorySnapshot:
        ctx = glyff.get_context()
        snapshot = await ctx.metadata.get(
            _METADATA_KEY,
            HistorySnapshot,
            execution_id=ctx.current_execution_id,
        )
        return snapshot if snapshot is not None else HistorySnapshot()

    async def save(self, snapshot: HistorySnapshot) -> None:
        ctx = glyff.get_context()
        # Long-lived runs do not commit their surrounding transaction.
        async with ctx.get_transaction_scope():
            await ctx.metadata.set(_METADATA_KEY, snapshot, HistorySnapshot)


class _History:
    """Mutable run history owned by the executor."""

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

    async def _load(self) -> None:
        snapshot = await self._storage.load()
        self._items = list(snapshot.items)
        self._completed_steps = snapshot.completed_steps

    def _current(self) -> list[HistoryItem]:
        return self._items

    async def _record_step(
        self, decision: HistoryItem, results: Sequence[HistoryItem]
    ) -> None:
        self._items.append(decision)
        self._items.extend(results)
        self._completed_steps += 1
        await self._storage.save(
            HistorySnapshot(tuple(self._items), self._completed_steps)
        )
