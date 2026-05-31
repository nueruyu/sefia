from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from sefia.event_publisher import EventPublisher
from sefia.executor import InferenceExecutor
from sefia.handlers.retry import RequestInferenceRetry
from sefia.interfaces import InferenceStrategy, ToolCollector
from sefia.models import (
    FinalAnswerDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
    ToolRegistry,
)


def sample_func(arg1: str) -> str:
    """Sample docstring."""
    return "implemented"


def sample_func_with_self(self, arg1: str) -> str:
    """Sample docstring."""
    return "implemented"


@pytest.fixture
def executor_dependencies(mocker: MockerFixture):
    """Provides a tuple of mocked dependencies for InferenceExecutor."""
    mock_strategy = mocker.AsyncMock(spec=InferenceStrategy)
    mock_collector = mocker.MagicMock(spec=ToolCollector)
    mock_publisher = mocker.AsyncMock(spec=EventPublisher)

    mock_collector.collect.return_value = ToolRegistry()

    def non_engrave(f):
        return f

    return (
        mock_strategy,
        mock_collector,
        mock_publisher,
        non_engrave,
    )


class TestInferenceExecutor:
    async def test_run_loop_with_tool_call_and_final_answer(
        self, executor_dependencies
    ):
        # Arrange
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies

        mock_strategy.decide_next_step.side_effect = [
            ToolCallDecision(
                calls=[ToolCallRequest(id="1", name="my_tool", arguments={"a": 1})]
            ),
            FinalAnswerDecision(answer="final result"),
        ]

        tool_registry = ToolRegistry()
        mock_tool_func = AsyncMock(return_value="tool result")
        tool_schema = {
            "type": "function",
            "function": {"name": "my_tool", "description": "", "parameters": {}},
        }
        tool_registry.add(mock_tool_func, tool_schema)
        mock_collector.collect.return_value = tool_registry

        executor = InferenceExecutor(
            sample_func_with_self,
            (object(), "value"),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        # Act
        result = await executor.run()

        # Assert
        assert result == "final result"
        assert mock_strategy.decide_next_step.call_count == 2
        mock_tool_func.assert_called_once_with(a=1)

    async def test_handles_nonexistent_tool_call(self, executor_dependencies):
        # Arrange
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.side_effect = [
            ToolCallDecision(
                calls=[ToolCallRequest(id="1", name="nonexistent_tool", arguments={})]
            ),
            FinalAnswerDecision(answer="recovered"),
        ]

        executor = InferenceExecutor(
            sample_func_with_self,
            (object(), "dummy_arg"),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        # Act
        result = await executor.run()

        # Assert
        assert result == "recovered"
        history = mock_strategy.decide_next_step.call_args_list[1].kwargs["history"]
        assert len(history) == 2
        assert isinstance(history[0], ToolCallDecision)
        assert isinstance(history[1], ToolCallResult)
        assert "Error: Tool 'nonexistent_tool' not found" in history[1].result

    async def test_raises_error_if_max_turns_exceeded(self, executor_dependencies):
        # Arrange
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies
        mock_strategy.decide_next_step.return_value = ToolCallDecision(calls=[])

        executor = InferenceExecutor(
            sample_func,
            ("dummy_arg",),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="Inference did not complete"):
            await executor.run()

    async def test_handles_request_inference_retry(self, executor_dependencies):
        # Arrange
        (
            mock_strategy,
            mock_collector,
            mock_publisher,
            non_engrave,
        ) = executor_dependencies

        executor = InferenceExecutor(
            sample_func,
            ("dummy_arg",),
            {},
            mock_strategy,
            mock_collector,
            non_engrave,
            mock_publisher,
        )
        executor._attempt_inference = AsyncMock(
            side_effect=[RequestInferenceRetry(), "attempt 2 succeeds"]
        )

        # Act
        result = await executor.run()

        # Assert
        assert result == "attempt 2 succeeds"
        assert executor._attempt_inference.call_count == 2
