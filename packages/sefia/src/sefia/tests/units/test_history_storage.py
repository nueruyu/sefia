from sefia import HistorySnapshot, HistoryStorage
from sefia._history import StepHistory
from sefia.inference import (
    HistoryItem,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)


class _InMemoryHistoryStorage(HistoryStorage):
    def __init__(self, initial: HistorySnapshot | None = None):
        self.snapshot = initial if initial is not None else HistorySnapshot()
        self.saves: list[HistorySnapshot] = []

    async def load(self) -> HistorySnapshot:
        return self.snapshot

    async def save(self, snapshot: HistorySnapshot) -> None:
        self.snapshot = snapshot
        self.saves.append(snapshot)


def _decision(i: int) -> ToolCallDecision:
    return ToolCallDecision(
        calls=[ToolCallRequest(id=str(i), name="a_tool", arguments={"i": i})]
    )


def _result(i: int) -> ToolCallResult:
    return ToolCallResult(tool_call_id=str(i), result=f"r{i}")


class TestStepHistory:
    async def test_loads_snapshot_into_items_and_step_count(self):
        storage = _InMemoryHistoryStorage(
            HistorySnapshot(items=(_decision(0), _result(0)), completed_steps=3)
        )
        store = StepHistory(storage)

        await store.load()

        assert list(store.items) == [_decision(0), _result(0)]
        assert store.completed_steps == 3

    async def test_items_is_an_immutable_snapshot(self):
        store = StepHistory(_InMemoryHistoryStorage())
        await store.record_step(_decision(0), [_result(0)])

        view = store.items
        assert isinstance(view, tuple)
        assert list(view) == [_decision(0), _result(0)]
        await store.record_step(_decision(1), [_result(1)])
        assert len(view) == 2  # the earlier snapshot is unaffected

    async def test_record_step_appends_and_persists_with_incremented_count(self):
        storage = _InMemoryHistoryStorage()
        store = StepHistory(storage)

        await store.record_step(_decision(0), [_result(0)])
        await store.record_step(_decision(1), [])

        assert store.completed_steps == 2
        assert [(s.items, s.completed_steps) for s in storage.saves] == [
            ((_decision(0), _result(0)), 1),
            ((_decision(0), _result(0), _decision(1)), 2),
        ]

    async def test_rewrite_persists_before_swapping_and_keeps_step_count(self):
        storage = _InMemoryHistoryStorage()
        store = StepHistory(storage)
        await store.record_step(_decision(0), [_result(0)])
        await store.record_step(_decision(1), [_result(1)])

        seen_in_memory: list[list[HistoryItem]] = []

        class OrderProbe(_InMemoryHistoryStorage):
            async def save(self, snapshot: HistorySnapshot) -> None:
                # In-memory items must still be the old content when save runs.
                seen_in_memory.append(list(store.items))
                await super().save(snapshot)

        store._storage = OrderProbe()
        await store.rewrite([_decision(1), _result(1)])

        assert seen_in_memory == [[_decision(0), _result(0), _decision(1), _result(1)]]
        assert list(store.items) == [_decision(1), _result(1)]
        assert store.completed_steps == 2
        assert store._storage.saves[-1].completed_steps == 2


class TestHistorySnapshot:
    def test_defaults_to_empty(self):
        snap = HistorySnapshot()
        assert snap.items == ()
        assert snap.completed_steps == 0
