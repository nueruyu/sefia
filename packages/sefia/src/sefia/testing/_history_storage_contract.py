"""Reusable pytest contract for ``HistoryStorage`` implementations."""

from .._interfaces.history_storage import HistorySnapshot, HistoryStorage
from ..inference import ToolCallResult


class HistoryStorageContract:
    """Shared persistence behavior required by an inference history store."""

    async def test_initial_snapshot_is_empty(
        self, history_storage: HistoryStorage
    ) -> None:
        assert await history_storage.load() == HistorySnapshot()

    async def test_snapshot_round_trips_and_can_be_replaced(
        self, history_storage: HistoryStorage
    ) -> None:
        first = HistorySnapshot(
            items=(ToolCallResult(tool_call_id="call-1", result="first"),),
            completed_steps=1,
        )
        second = HistorySnapshot(
            items=(ToolCallResult(tool_call_id="call-2", result="second"),),
            completed_steps=2,
        )

        await history_storage.save(first)
        assert await history_storage.load() == first

        await history_storage.save(second)
        assert await history_storage.load() == second


__all__ = ["HistoryStorageContract"]
