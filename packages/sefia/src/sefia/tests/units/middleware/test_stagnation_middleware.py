from datetime import datetime

import pytest

from sefia._interfaces.middleware import StepContext
from sefia.inference import FinalAnswerDecision, ToolCallDecision, ToolCallRequest
from sefia.middleware import StagnationDetector, StagnationError


async def _step(middleware: StagnationDetector, name: str, args: dict, step: int = 0):
    """Drives one step whose decision calls a single tool."""
    decision = ToolCallDecision(
        calls=[ToolCallRequest(id="1", name=name, arguments=args)]
    )

    async def nxt():
        return decision

    return await middleware.wrap(StepContext(step=step, history=[]), nxt)


class TestStagnationDetector:
    def test_rejects_invalid_max_repeats(self):
        # A limit of 1 would flag the first tool call, so values < 2 are invalid.
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
        await _step(middleware, "test_tool", {"a": 2}, step=1)  # Should not raise

    async def test_history_resets_after_different_call(self):
        middleware = StagnationDetector(max_repeats=2)

        await _step(middleware, "tool1", {"a": 1}, step=0)
        await _step(middleware, "tool2", {"b": 2}, step=1)
        await _step(middleware, "tool1", {"a": 1}, step=2)  # history broken by tool2

    async def test_does_not_raise_if_limit_not_reached(self):
        middleware = StagnationDetector(max_repeats=3)

        await _step(middleware, "test_tool", {"a": 1}, step=0)
        await _step(middleware, "test_tool", {"a": 1}, step=1)  # Should not raise

    async def test_history_clears_on_new_attempt(self):
        # The same middleware instance is reused across retried attempts. A new
        # attempt restarts at step 0 and must not inherit the previous attempt's
        # tool-call history, or it would raise a false-positive StagnationError.
        middleware = StagnationDetector(max_repeats=2)

        await _step(middleware, "test_tool", {"a": 1}, step=0)  # first attempt
        await _step(middleware, "test_tool", {"a": 1}, step=0)  # new attempt; ok

    async def test_ignores_final_answer_decisions(self):
        middleware = StagnationDetector(max_repeats=2)

        async def nxt():
            return FinalAnswerDecision(answer="done")

        decision = await middleware.wrap(StepContext(step=0, history=[]), nxt)
        assert isinstance(decision, FinalAnswerDecision)

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
            await middleware.wrap(StepContext(step=0, history=[]), nxt)

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
