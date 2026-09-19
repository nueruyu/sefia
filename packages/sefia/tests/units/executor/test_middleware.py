from collections.abc import Awaitable, Callable
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from sefia import (
    DecisionContext,
    DecisionMiddleware,
    events,
    InferenceMiddleware,
    StepContext,
    StepMiddleware,
)
from sefia._executor import InferenceExecutor, _compose
from sefia.event_system import Event
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


async def test_inference_middleware_retries_inference_failure(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    retry_once: InferenceMiddleware,
) -> None:
    mock_strategy.decide_next_step.side_effect = [
        ValueError("flaky inference"),
        ResultDecision(result="second attempt"),
    ]

    executor = make_executor(
        inference_middlewares=[retry_once],
    )

    result = await executor.run()

    assert result == "second attempt"
    assert mock_strategy.decide_next_step.call_count == 2


async def test_failure_from_reentered_attempt_propagates(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    retry_once: InferenceMiddleware,
) -> None:
    mock_strategy.decide_next_step.side_effect = ValueError("always flaky")

    executor = make_executor(
        inference_middlewares=[retry_once],
    )

    with pytest.raises(ValueError, match="always flaky"):
        await executor.run()

    # The second failure propagates out of middleware.
    assert mock_strategy.decide_next_step.call_count == 2


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


async def test_decision_middlewares_wrap_each_durable_step_in_declared_order(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    mock_publisher: AsyncMock,
) -> None:
    calls: list[str] = []
    contexts: list[DecisionContext] = []
    decisions = [ToolCallsDecision([]), ResultDecision("done")]

    @final
    class RecordDecision(DecisionMiddleware):
        def __init__(self, label: str) -> None:
            self.label = label

        @override
        async def wrap(
            self, ctx: DecisionContext, nxt: Callable[[], Awaitable[StepDecision]]
        ) -> StepDecision:
            contexts.append(ctx)
            calls.append(f"{self.label}:enter:{ctx.step}")
            decision = await nxt()
            assert decision is decisions[ctx.step]
            calls.append(f"{self.label}:exit:{ctx.step}")
            return decision

    @final
    class RecordStep(StepMiddleware):
        @override
        async def wrap(
            self, ctx: StepContext, nxt: Callable[[], Awaitable[StepDecision]]
        ) -> StepDecision:
            calls.append(f"step:enter:{ctx.step}")
            decision = await nxt()
            calls.append(f"step:exit:{ctx.step}")
            return decision

    def engrave(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapped(step: int) -> StepDecision:
            calls.append(f"{name}:enter:{step}")
            decision = await function(step)
            calls.append(f"{name}:exit:{step}")
            return decision

        return wrapped

    async def decide(**kwargs: Any) -> StepDecision:
        ctx = contexts[-1]
        assert kwargs["function_info"] is ctx.function_info
        assert kwargs["history"] is ctx.history
        calls.append(f"strategy:{ctx.step}")
        return decisions[ctx.step]

    async def publish(event: Event) -> None:
        if isinstance(event, (events.BeforeInferenceStep, events.AfterInferenceStep)):
            calls.append(type(event).__name__)

    mock_strategy.decide_next_step.side_effect = decide
    mock_publisher.publish.side_effect = publish
    executor = make_executor(
        engrave=engrave,
        step_middlewares=[RecordStep()],
        decision_middlewares=[RecordDecision("outer"), RecordDecision("inner")],
    )

    assert await executor.run() == "done"
    assert calls == [
        entry
        for step in (0, 1)
        for entry in (
            f"step:enter:{step}",
            f"inference.step:enter:{step}",
            "BeforeInferenceStep",
            f"outer:enter:{step}",
            f"inner:enter:{step}",
            f"strategy:{step}",
            f"inner:exit:{step}",
            f"outer:exit:{step}",
            "AfterInferenceStep",
            f"inference.step:exit:{step}",
            f"step:exit:{step}",
        )
    ]
    assert [ctx.step for ctx in contexts] == [0, 0, 1, 1]
    assert all(ctx.function_info is executor.func_info for ctx in contexts)
    assert contexts[0].history == ()
    assert contexts[2].history == (decisions[0],)
    assert all(isinstance(ctx.history, tuple) for ctx in contexts)


@pytest.mark.parametrize(
    "replacement", [ResultDecision("transformed"), ToolCallsDecision([])]
)
async def test_decision_middleware_transforms_the_executor_and_event_decision(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    mock_publisher: AsyncMock,
    replacement: StepDecision,
) -> None:
    original = ResultDecision("original")
    mock_strategy.decide_next_step.return_value = original

    @final
    class Transform(DecisionMiddleware):
        @override
        async def wrap(
            self, ctx: DecisionContext, nxt: Callable[[], Awaitable[StepDecision]]
        ) -> StepDecision:
            assert await nxt() is original
            return replacement if ctx.step == 0 else ResultDecision("done")

    result = await make_executor(decision_middlewares=[Transform()]).run()

    assert result == (
        "transformed" if isinstance(replacement, ResultDecision) else "done"
    )
    after = [
        call.args[0]
        for call in mock_publisher.publish.call_args_list
        if isinstance(call.args[0], events.AfterInferenceStep)
    ]
    assert after[0].decision is replacement
    assert mock_strategy.decide_next_step.await_count == len(after)


async def test_decision_middleware_can_short_circuit(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    mock_publisher: AsyncMock,
) -> None:
    decision = ResultDecision("short circuit")

    @final
    class ShortCircuit(DecisionMiddleware):
        @override
        async def wrap(
            self, ctx: DecisionContext, nxt: Callable[[], Awaitable[StepDecision]]
        ) -> StepDecision:
            return decision

    assert (
        await make_executor(decision_middlewares=[ShortCircuit()]).run()
        == "short circuit"
    )
    mock_strategy.decide_next_step.assert_not_called()
    after = [
        call.args[0]
        for call in mock_publisher.publish.call_args_list
        if isinstance(call.args[0], events.AfterInferenceStep)
    ]
    assert len(after) == 1
    assert after[0].decision is decision


@pytest.mark.parametrize("invalid_return", [False, True])
async def test_decision_middleware_failure_is_reported_before_step_returns(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    mock_publisher: AsyncMock,
    invalid_return: bool,
) -> None:
    error = ValueError("rejected")
    mock_strategy.decide_next_step.return_value = ResultDecision("original")

    @final
    class Reject(DecisionMiddleware):
        @override
        async def wrap(
            self, ctx: DecisionContext, nxt: Callable[[], Awaitable[StepDecision]]
        ) -> StepDecision:
            await nxt()
            if invalid_return:
                return cast(StepDecision, "invalid")
            raise error

    executor = make_executor(decision_middlewares=[Reject()])
    expected_error = TypeError if invalid_return else ValueError
    with pytest.raises(
        expected_error, match="Unknown decision type" if invalid_return else "rejected"
    ) as caught:
        await executor._next_step(0)

    published = [call.args[0] for call in mock_publisher.publish.call_args_list]
    assert [type(event) for event in published] == [
        events.BeforeInferenceStep,
        events.InferenceStepFailed,
    ]
    failure = published[-1]
    assert failure.error is caught.value
    if not invalid_return:
        assert caught.value is error


async def test_decision_middleware_can_call_next_again(
    make_executor: Callable[..., InferenceExecutor], mock_strategy: AsyncMock
) -> None:
    mock_strategy.decide_next_step.side_effect = [
        ResultDecision("first"),
        ResultDecision("second"),
    ]

    @final
    class Repeat(DecisionMiddleware):
        @override
        async def wrap(
            self, ctx: DecisionContext, nxt: Callable[[], Awaitable[StepDecision]]
        ) -> StepDecision:
            await nxt()
            return await nxt()

    assert await make_executor(decision_middlewares=[Repeat()]).run() == "second"
    assert mock_strategy.decide_next_step.await_count == 2
