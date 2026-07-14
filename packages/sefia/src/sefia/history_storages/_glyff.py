import glyff

from .._interfaces.history_storage import HistorySnapshot, HistoryStorage

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
