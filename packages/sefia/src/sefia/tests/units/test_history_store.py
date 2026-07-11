from collections.abc import Sequence

from sefia import HistoryStore, StepContext, TransientHistoryStore
from sefia.inference import (
    HistoryItem,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)


class _RecordingHistoryStore(HistoryStore):
    def __init__(self, initial: list[HistoryItem] | None = None):
        self.items: list[HistoryItem] = list(initial or [])
        self.saves: list[list[HistoryItem]] = []

    async def load(self) -> list[HistoryItem]:
        return list(self.items)

    async def save(self, items: Sequence[HistoryItem]) -> None:
        self.items = list(items)
        self.saves.append(list(items))


def _sample_history() -> list[HistoryItem]:
    decision = ToolCallDecision(
        calls=[ToolCallRequest(id="1", name="a_tool", arguments={"x": 1})]
    )
    return [decision, ToolCallResult(tool_call_id="1", result="ok")]


class TestTransientHistoryStore:
    async def test_load_is_empty_and_save_is_a_noop(self):
        store = TransientHistoryStore()
        await store.save(_sample_history())
        assert await store.load() == []


class TestStepContextRewriteHistory:
    async def test_rewrites_the_shared_list_in_place(self):
        # The executor holds the same list object; the rewrite must mutate it
        # in place so the loop (and other middleware) see the new content.
        history = _sample_history()
        ctx = StepContext(step=1, history=history)

        await ctx.rewrite_history(history[-1:])

        assert ctx.history is history
        assert history == [ToolCallResult(tool_call_id="1", result="ok")]

    async def test_persists_to_the_store_before_mutating(self):
        history = _sample_history()
        observed_at_save: list[list[HistoryItem]] = []

        class OrderProbe(_RecordingHistoryStore):
            async def save(self, items: Sequence[HistoryItem]) -> None:
                # The in-memory history must still be the old content when the
                # store commits, so a failed save loses nothing.
                observed_at_save.append(list(history))
                await super().save(items)

        store = OrderProbe()
        ctx = StepContext(step=1, history=history, history_store=store)

        await ctx.rewrite_history([])

        assert observed_at_save == [_sample_history()]
        assert store.items == []
        assert history == []
