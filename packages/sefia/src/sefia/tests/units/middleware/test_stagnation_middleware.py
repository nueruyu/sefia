from datetime import datetime

import pytest

from sefia.interfaces.middleware import StepContext
from sefia.middleware.signals import StagnationError
from sefia.middleware.stagnation import StagnationMiddleware
from sefia.models import (
    FinalAnswerDecision,
    ToolCallDecision,
    ToolCallRequest,
)


def _ctx() -> StepContext:
    return StepContext(step=0, history=[])


async def _step(middleware: StagnationMiddleware, name: str, args: dict):
    """Drives one step whose decision calls a single tool."""
    decision = ToolCallDecision(
        calls=[ToolCallRequest(id="1", name=name, arguments=args)]
    )

    async def nxt():
        return decision

    return await middleware.wrap(_ctx(), nxt)


class TestStagnationMiddleware:
    async def test_raises_error_on_repeated_calls(self):
        middleware = StagnationMiddleware(max_repeats=3)

        await _step(middleware, "test_tool", {"a": 1})
        await _step(middleware, "test_tool", {"a": 1})

        with pytest.raises(StagnationError):
            await _step(middleware, "test_tool", {"a": 1})

    async def test_does_not_raise_for_different_calls(self):
        middleware = StagnationMiddleware(max_repeats=2)

        await _step(middleware, "test_tool", {"a": 1})
        await _step(middleware, "test_tool", {"a": 2})  # Should not raise

    async def test_history_resets_after_different_call(self):
        middleware = StagnationMiddleware(max_repeats=2)

        await _step(middleware, "tool1", {"a": 1})
        await _step(middleware, "tool2", {"b": 2})
        await _step(middleware, "tool1", {"a": 1})  # history broken by tool2

    async def test_does_not_raise_if_limit_not_reached(self):
        middleware = StagnationMiddleware(max_repeats=3)

        await _step(middleware, "test_tool", {"a": 1})
        await _step(middleware, "test_tool", {"a": 1})  # Should not raise

    async def test_ignores_final_answer_decisions(self):
        middleware = StagnationMiddleware(max_repeats=1)

        async def nxt():
            return FinalAnswerDecision(answer="done")

        decision = await middleware.wrap(_ctx(), nxt)
        assert isinstance(decision, FinalAnswerDecision)

    async def test_records_each_call_in_a_multi_call_decision(self):
        middleware = StagnationMiddleware(max_repeats=3)
        decision = ToolCallDecision(
            calls=[
                ToolCallRequest(id=str(i), name="t", arguments={"a": 1})
                for i in range(3)
            ]
        )

        async def nxt():
            return decision

        with pytest.raises(StagnationError):
            await middleware.wrap(_ctx(), nxt)

    def test_hashes_nested_dictionaries_consistently(self):
        middleware = StagnationMiddleware()
        args1 = {"a": 1, "nested": {"c": 3, "b": 2}}
        args2 = {"nested": {"b": 2, "c": 3}, "a": 1}

        assert middleware._hash_tool_call(
            "test_tool", args1
        ) == middleware._hash_tool_call("test_tool", args2)

    def test_handles_non_serializable_types_with_fallback(self):
        middleware = StagnationMiddleware()
        now = datetime.now()

        h = middleware._hash_tool_call("test_tool", {"dt": now, "num": 1})
        assert str(now) in h
