import pytest
from sefia import HistorySnapshot, HistoryStorage, HistoryStore, StepContext
from sefia.inference import (
    HistoryItem,
    ResultDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)

from sefios.middleware import HistoryCompactor, truncate_history


def _step(i: int) -> list[HistoryItem]:
    """One completed step: a decision and its tool result."""
    return [
        ToolCallDecision(
            calls=[ToolCallRequest(id=str(i), name="a_tool", arguments={"i": i})]
        ),
        ToolCallResult(tool_call_id=str(i), result=f"result {i}"),
    ]


def _history(steps: int) -> list[HistoryItem]:
    return [item for i in range(steps) for item in _step(i)]


class _InMemoryHistoryStorage(HistoryStorage):
    def __init__(self, initial: HistorySnapshot | None = None):
        self.snapshot = initial if initial is not None else HistorySnapshot()
        self.saves: list[HistorySnapshot] = []

    async def load(self) -> HistorySnapshot:
        return self.snapshot

    async def save(self, snapshot: HistorySnapshot) -> None:
        self.snapshot = snapshot
        self.saves.append(snapshot)


async def _history_store(
    items: list[HistoryItem],
) -> tuple[HistoryStore, _InMemoryHistoryStorage]:
    storage = _InMemoryHistoryStorage(
        HistorySnapshot(items=tuple(items), completed_steps=len(items) // 2)
    )
    store = HistoryStore(storage)
    await store._load()
    return store, storage


async def _nxt() -> ResultDecision:
    return ResultDecision(result="done")


class TestTruncateHistory:
    def test_keeps_short_history_untouched(self):
        history = _history(2)
        assert truncate_history(history, keep_items=4) == history

    def test_drops_orphaned_leading_results(self):
        # Cutting at 3 items would start the tail with step 1's result, whose
        # decision was cut off — the orphan is dropped too.
        history = _history(2)
        assert truncate_history(history, keep_items=3) == _step(1)

    def test_keeps_tail_starting_at_a_decision(self):
        history = _history(3)
        assert truncate_history(history, keep_items=4) == _history(3)[2:]

    def test_zero_keep_items_empties_the_history(self):
        assert truncate_history(_history(2), keep_items=0) == []


class TestHistoryCompactor:
    async def test_does_not_compact_under_the_threshold(self):
        store, storage = await _history_store(_history(2))
        ctx = StepContext(step=2, history=store)

        decision = await HistoryCompactor(max_items=4).wrap(ctx, _nxt)

        assert decision == ResultDecision(result="done")
        assert list(store.items) == _history(2)
        assert storage.saves == []

    async def test_compacts_and_persists_before_the_step(self):
        store, storage = await _history_store(_history(3))
        ctx = StepContext(step=3, history=store)

        await HistoryCompactor(max_items=5, keep_items=2).wrap(ctx, _nxt)

        assert list(store.items) == _step(2)
        # Persisted, and the step count is preserved (compaction is not a step).
        assert len(storage.saves) == 1
        assert storage.saves[0].completed_steps == 3

    async def test_uses_a_custom_async_compactor(self):
        store, _ = await _history_store(_history(3))
        seen: list[list[HistoryItem]] = []

        async def summarize(items: list[HistoryItem]) -> list[HistoryItem]:
            seen.append(items)
            return [ToolCallResult(tool_call_id="summary", result="3 steps ran")]

        ctx = StepContext(step=3, history=store)
        await HistoryCompactor(max_items=5, compact=summarize).wrap(ctx, _nxt)

        assert seen == [_history(3)]
        assert list(store.items) == [
            ToolCallResult(tool_call_id="summary", result="3 steps ran")
        ]

    async def test_rejects_conflicting_configuration(self):
        with pytest.raises(ValueError, match="not both"):
            HistoryCompactor(max_items=4, keep_items=2, compact=lambda h: h)
        with pytest.raises(ValueError, match="max_items"):
            HistoryCompactor(max_items=0)
