from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest
from sefia import (
    InferenceMiddleware,
    events,
)
from sefia._executor import InferenceExecutor
from sefia.events import AttemptStart, StepStarted
from sefia.inference import (
    ResultDecision,
    ToolCallsDecision,
)


async def test_fires_step_started_for_every_step(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    mock_publisher: AsyncMock,
) -> None:
    mock_strategy.decide_next_step.side_effect = [
        ToolCallsDecision(calls=[]),
        ResultDecision(result="done"),
    ]

    executor = make_executor()

    await executor.run()

    step_events = [
        call.args[0]
        for call in mock_publisher.publish.call_args_list
        if isinstance(call.args[0], StepStarted)
    ]
    assert [event.step for event in step_events] == [0, 1]


async def test_failed_step_publishes_event_and_reraises_by_default(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    mock_publisher: AsyncMock,
) -> None:
    error = ValueError("boom")
    mock_strategy.decide_next_step.side_effect = error

    executor = make_executor()

    with pytest.raises(ValueError, match="boom"):
        await executor.run()

    published = [call.args[0] for call in mock_publisher.publish.call_args_list]
    step_failures = [e for e in published if isinstance(e, events.InferenceStepFailed)]
    assert len(step_failures) == 1
    assert step_failures[0].error is error


async def test_recoverable_inference_error_yields_without_failing_run(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    mock_publisher: AsyncMock,
) -> None:
    from sefia.exceptions import InvalidInferenceResponseError

    error = InvalidInferenceResponseError("malformed response")
    mock_strategy.decide_next_step.side_effect = error

    executor = make_executor()

    with pytest.raises(InvalidInferenceResponseError, match="malformed response"):
        await executor.run()

    published = [call.args[0] for call in mock_publisher.publish.call_args_list]
    step_failures = [e for e in published if isinstance(e, events.InferenceStepFailed)]
    assert len(step_failures) == 1
    assert step_failures[0].error is error
    # The run itself is not reported as failed — it is a recoverable yield.
    assert not any(isinstance(e, events.InferenceFailed) for e in published)


async def test_retry_middleware_publishes_attempt_start_per_attempt(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    mock_publisher: AsyncMock,
    retry_once: InferenceMiddleware,
) -> None:
    mock_strategy.decide_next_step.side_effect = [
        ValueError("flaky inference"),
        ResultDecision(result="attempt 2 succeeds"),
    ]

    executor = make_executor(
        inference_middlewares=[retry_once],
    )
    result = await executor.run()

    assert result == "attempt 2 succeeds"
    attempt_events = [
        call.args[0]
        for call in mock_publisher.publish.call_args_list
        if isinstance(call.args[0], AttemptStart)
    ]
    assert len(attempt_events) == 2
