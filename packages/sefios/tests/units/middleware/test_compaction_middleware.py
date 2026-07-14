import pytest
from sefia import StepContext
from sefia._history import StepHistory
from sefia.inference import (
    HistoryItem,
    ResultDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)

from sefios.middleware import HistoryCompactor
from sefios.middleware._compaction import _truncate_history


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


async def _nxt() -> ResultDecision:
    return ResultDecision(result="done")


class TestTruncateHistory:
    def test_keeps_short_history_untouched(self):
        history = _history(2)
        assert _truncate_history(history, keep_items=4) == history

    def test_drops_orphaned_leading_results(self):
        # Cutting at 3 items would start the tail with step 1's result, whose
        # decision was cut off — the orphan is dropped too.
        history = _history(2)
        assert _truncate_history(history, keep_items=3) == _step(1)

    def test_keeps_tail_starting_at_a_decision(self):
        history = _history(3)
        assert _truncate_history(history, keep_items=4) == _history(3)[2:]

    def test_zero_keep_items_empties_the_history(self):
        assert _truncate_history(_history(2), keep_items=0) == []


class TestHistoryCompactor:
    async def test_does_not_compact_under_the_threshold(self):
        history = StepHistory(_history(2))
        ctx = StepContext(step=2, history=history)

        decision = await HistoryCompactor(max_items=4).wrap(ctx, _nxt)

        assert decision == ResultDecision(result="done")
        assert list(history.items) == _history(2)

    async def test_compacts_the_history_before_the_step(self):
        history = StepHistory(_history(3))
        ctx = StepContext(step=3, history=history)

        await HistoryCompactor(max_items=5, keep_items=2).wrap(ctx, _nxt)

        assert list(history.items) == _step(2)

    async def test_rejects_invalid_configuration(self):
        with pytest.raises(ValueError, match="max_items"):
            HistoryCompactor(max_items=0)
