from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture
from sefia import (
    InferenceMiddleware,
    ToolRegistry,
)
from sefia._executor import InferenceExecutor
from sefia.inference import (
    ResultDecision,
    ToolCallResult,
    ToolCallsDecision,
)
from sefia.testing import make_tool_call_request


async def test_rejects_unknown_decision_from_custom_strategy(
    make_executor: Callable[..., InferenceExecutor], mock_strategy: AsyncMock
) -> None:
    mock_strategy.decide_next_step.return_value = object()

    executor = make_executor()

    with pytest.raises(TypeError, match="Unknown decision type"):
        await executor.run()


async def test_run_loop_with_tool_call_and_result(
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    mock_collector: MagicMock,
) -> None:
    mock_strategy.decide_next_step.side_effect = [
        ToolCallsDecision(
            calls=[make_tool_call_request(id="1", name="my_tool", arguments={"a": 1})]
        ),
        ResultDecision(result="final result"),
    ]

    tool_registry = ToolRegistry()
    mock_tool_func = AsyncMock(return_value="tool result")
    tool_registry.add(mock_tool_func, name="my_tool")
    mock_collector.collect.return_value = tool_registry

    executor = make_executor()

    result = await executor.run()

    assert result == "final result"
    assert mock_strategy.decide_next_step.call_count == 2
    mock_tool_func.assert_called_once_with(a=1)


async def test_tool_result_is_fed_back_without_restarting(
    mocker: MockerFixture,
    make_executor: Callable[..., InferenceExecutor],
    mock_strategy: AsyncMock,
    retry_once: InferenceMiddleware,
) -> None:
    mock_strategy.decide_next_step.side_effect = [
        ToolCallsDecision(
            calls=[make_tool_call_request(id="1", name="boom_tool", arguments={})]
        ),
        ResultDecision(result="recovered"),
    ]

    tool_result = ToolCallResult(tool_call_id="1", result="tool failure")
    mocker.patch("sefia._executor.call_tools", return_value=[tool_result])

    executor = make_executor(
        inference_middlewares=[retry_once],
    )

    result = await executor.run()

    assert result == "recovered"
    # No retry: the strategy was consulted exactly twice (tool call, then
    # the result informed by the error).
    assert mock_strategy.decide_next_step.call_count == 2
    history = mock_strategy.decide_next_step.call_args_list[1].kwargs["history"]
    assert isinstance(history[-1], ToolCallResult)
    assert history[-1] is tool_result


async def test_internal_calls_use_stable_names_and_step_index(
    make_executor: Callable[..., InferenceExecutor], mock_strategy: AsyncMock
) -> None:
    mock_strategy.decide_next_step.side_effect = [
        ToolCallsDecision(calls=[]),
        ResultDecision(result="done"),
    ]

    engraved_names: list[str] = []
    engraved_step_args: list[tuple[Any, ...]] = []

    def recording_engrave(
        name: str, f: Callable[..., Awaitable[Any]]
    ) -> Callable[..., Awaitable[Any]]:
        engraved_names.append(name)

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if name == "inference.step":
                engraved_step_args.append(args)
            return await f(*args, **kwargs)

        return wrapper

    executor = make_executor(
        engrave=recording_engrave,
    )

    await executor.run()

    # Two steps, engraved on their indices (0, 1), not on a history list.
    assert engraved_names == ["inference.step", "inference.tool_calls"]
    assert engraved_step_args == [(0,), (1,)]
