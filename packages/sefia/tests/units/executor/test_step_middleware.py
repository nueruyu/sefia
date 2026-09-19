from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock

import pytest
from sefia import (
    StepContext,
    StepMiddleware,
)
from sefia._executor import InferenceExecutor, _compose
from sefia.exceptions import PauseException
from sefia.inference import (
    ResultDecision,
    StepDecision,
    ToolCallsDecision,
)
from sefia.testing import make_step_context
from typing_extensions import final, override


async def test_step_middleware_can_stop_the_loop(
    make_executor: Callable[..., InferenceExecutor], mock_strategy: AsyncMock
) -> None:
    @final
    class Stop(StepMiddleware):
        @override
        async def wrap(
            self, ctx: StepContext, nxt: Callable[[], Awaitable[StepDecision]]
        ) -> StepDecision:
            raise PauseException("stopped")

    with pytest.raises(PauseException, match="stopped"):
        await make_executor(step_middlewares=[Stop()]).run()
    mock_strategy.decide_next_step.assert_not_called()


async def test_step_middlewares_compose_in_declared_order() -> None:
    calls: list[str] = []

    @final
    class Recorder(StepMiddleware):
        def __init__(self, label: str):
            self.label = label

        @override
        async def wrap(
            self, ctx: StepContext, nxt: Callable[[], Awaitable[StepDecision]]
        ) -> StepDecision:
            calls.append(f"{self.label}:enter:{ctx.step}")
            decision = await nxt()
            calls.append(f"{self.label}:exit:{ctx.step}")
            return decision

    result = ResultDecision(result="done")
    core = AsyncMock(return_value=result)
    for step in (0, 1):
        chain = _compose(
            [Recorder("outer"), Recorder("inner")], make_step_context(step=step), core
        )
        assert await chain() is result
    assert core.await_count == 2
    assert calls == [
        "outer:enter:0",
        "inner:enter:0",
        "inner:exit:0",
        "outer:exit:0",
        "outer:enter:1",
        "inner:enter:1",
        "inner:exit:1",
        "outer:exit:1",
    ]


async def test_executor_applies_middlewares_in_order_on_every_step(
    make_executor: Callable[..., InferenceExecutor], mock_strategy: AsyncMock
) -> None:
    calls: list[tuple[str, int]] = []

    @final
    class Record(StepMiddleware):
        def __init__(self, label: str) -> None:
            self.label = label

        @override
        async def wrap(
            self, ctx: StepContext, nxt: Callable[[], Awaitable[StepDecision]]
        ) -> StepDecision:
            calls.append((self.label, ctx.step))
            return await nxt()

    mock_strategy.decide_next_step.side_effect = [
        ToolCallsDecision([]),
        ResultDecision("done"),
    ]
    result = await make_executor(
        step_middlewares=[Record("first"), Record("second")]
    ).run()

    assert result == "done"
    assert calls == [("first", 0), ("second", 0), ("first", 1), ("second", 1)]
