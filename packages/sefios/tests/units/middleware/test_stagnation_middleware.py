from datetime import datetime

import pytest
from sefia import HistorySnapshot, HistoryStorage, StepContext
from sefia._history import StepHistory
from sefia.inference import ResultDecision, ToolCallDecision, ToolCallRequest
from sefios.middleware import StagnationDetector, StagnationError


class _NoHistory(HistoryStorage):
    async def load(self) -> HistorySnapshot:
        return HistorySnapshot()

    async def save(self, snapshot: HistorySnapshot) -> None:
        pass


def _empty_history() -> StepHistory:
    return StepHistory(_NoHistory())


async def _step(middleware: StagnationDetector, name: str, args: dict, step: int = 0):
    """Drives one step whose decision calls a single tool."""
    decision = ToolCallDecision(
        calls=[ToolCallRequest(id="1", name=name, arguments=args)]
    )

    async def nxt():
        return decision

    return await middleware.wrap(StepContext(step=step, history=_empty_history()), nxt)


class TestStagnationDetector:
    def test_rejects_invalid_max_repeats(self):
        for invalid in (0, 1):
            with pytest.raises(ValueError):
                StagnationDetector(max_repeats=invalid)

    async def test_raises_error_on_repeated_calls(self):
        middleware = StagnationDetector(max_repeats=3)

        await _step(middleware, "test_tool", {"a": 1}, step=0)
        await _step(middleware, "test_tool", {"a": 1}, step=1)

        with pytest.raises(StagnationError):
            await _step(middleware, "test_tool", {"a": 1}, step=2)

    async def test_does_not_raise_for_different_calls(self):
        middleware = StagnationDetector(max_repeats=2)

        await _step(middleware, "test_tool", {"a": 1}, step=0)
        await _step(middleware, "test_tool", {"a": 2}, step=1)

    async def test_history_resets_after_different_call(self):
        middleware = StagnationDetector(max_repeats=2)

        await _step(middleware, "tool1", {"a": 1}, step=0)
        await _step(middleware, "tool2", {"b": 2}, step=1)
        await _step(middleware, "tool1", {"a": 1}, step=2)

    async def test_does_not_raise_if_limit_not_reached(self):
        middleware = StagnationDetector(max_repeats=3)

        await _step(middleware, "test_tool", {"a": 1}, step=0)
        await _step(middleware, "test_tool", {"a": 1}, step=1)

    async def test_history_clears_on_new_attempt(self):
        middleware = StagnationDetector(max_repeats=2)

        await _step(middleware, "test_tool", {"a": 1}, step=0)
        await _step(middleware, "test_tool", {"a": 1}, step=0)

    async def test_ignores_result_decisions(self):
        middleware = StagnationDetector(max_repeats=2)

        async def nxt():
            return ResultDecision(result="done")

        decision = await middleware.wrap(
            StepContext(step=0, history=_empty_history()), nxt
        )
        assert isinstance(decision, ResultDecision)

    async def test_records_each_call_in_a_multi_call_decision(self):
        middleware = StagnationDetector(max_repeats=3)
        decision = ToolCallDecision(
            calls=[
                ToolCallRequest(id=str(i), name="t", arguments={"a": 1})
                for i in range(3)
            ]
        )

        async def nxt():
            return decision

        with pytest.raises(StagnationError):
            await middleware.wrap(StepContext(step=0, history=_empty_history()), nxt)

    def test_hashes_nested_dictionaries_consistently(self):
        middleware = StagnationDetector()
        args1 = {"a": 1, "nested": {"c": 3, "b": 2}}
        args2 = {"nested": {"b": 2, "c": 3}, "a": 1}

        assert middleware._hash_tool_call(
            "test_tool", args1
        ) == middleware._hash_tool_call("test_tool", args2)

    def test_handles_non_serializable_types_with_fallback(self):
        middleware = StagnationDetector()
        now = datetime.now()

        h = middleware._hash_tool_call("test_tool", {"dt": now, "num": 1})
        assert str(now) in h
