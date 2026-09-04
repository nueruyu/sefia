import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from sefia._tool_system import ToolRegistry
from sefia.event_system import Event, EventPublisher
from sefia.exceptions import InvalidInferenceResponseError
from sefia.inference import (
    FunctionInfo,
    ResultDecision,
    ToolCallsDecision,
    ToolCallRequest,
    ToolCallResult,
)
from sefia.llm import LLMInferenceStrategy, LLMResponse, MarkdownPromptRenderer
from sefia.llm.exceptions import LLMResponseDecodingError
from sefia.llm.events import (
    LLMReasoningTokenReceived,
    LLMResponseRepairAttempt,
    LLMTokenReceived,
)
from sefia.llm.llm_output import LLMOutput
from sefia.llm.transports import StructuredDecisionTransport
from sefia.pydantic import PydanticModelBackend
from sefia.pydantic._json_utils import pydantic_json_default


class MockEventPublisher(EventPublisher):
    def __init__(self):
        super().__init__(handlers=[])
        self.events: list[Event] = []

    async def publish(self, event: Event) -> None:
        self.events.append(event)


@dataclass(frozen=True)
class MyOutput:
    name: str
    value: int


@dataclass(frozen=True)
class MyIssue:
    description: str


def search(q: str) -> str:
    """Search for something."""
    raise NotImplementedError


def my_tool(param: int) -> str:
    """A tool taking a single integer parameter."""
    raise NotImplementedError


def chat_tool() -> str:
    """A tool taking no arguments."""
    raise NotImplementedError


_BACKEND = PydanticModelBackend()


def _tool_registry(*funcs: Callable[..., Any]) -> ToolRegistry:
    registry = ToolRegistry()
    for func in funcs:
        registry.add(func, name=_BACKEND.tool_name(func))
    return registry


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    return AsyncMock()


def _make_strategy(
    llm_client: Any = None, *, stream: bool = False, max_repair_attempts: int = 2
) -> LLMInferenceStrategy:
    """The strategy under test, with a stub prompt renderer."""
    client = llm_client if llm_client is not None else AsyncMock()
    return LLMInferenceStrategy(
        llm_client=client,
        result_format_factory=PydanticModelBackend(),
        prompt_renderer=MarkdownPromptRenderer(json_default=pydantic_json_default),
        decision_transport=StructuredDecisionTransport(),
        stream=stream,
        max_repair_attempts=max_repair_attempts,
    )


def _structured_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        structured_output=LLMOutput.parse_json(content),
    )


def _function_info(
    return_type: Any = str,
    arguments: dict[str, Any] | None = None,
    type_hints: dict[str, Any] | None = None,
    instructions: str = "instructions",
) -> FunctionInfo:
    return FunctionInfo(
        qualname="test",
        instructions=instructions,
        bound_arguments=arguments or {},
        type_hints=type_hints or {},
        return_type=return_type,
        args=(),
        kwargs={},
    )


class TestLLMInferenceStrategy:
    async def test_decide_next_step_handles_tool_calls(
        self, mock_llm_client: AsyncMock
    ):
        tool_calls_payload = json.dumps(
            {
                "decision": "tool_calls",
                "tool_calls": [{"name": "my_tool", "arguments": {"param": 1}}],
            }
        )
        mock_llm_client.complete.return_value = _structured_response(tool_calls_payload)
        strategy = _make_strategy(mock_llm_client, stream=True)

        decision = await strategy.decide_next_step(
            _function_info(instructions="do it"),
            [],
            _tool_registry(my_tool),
            MockEventPublisher(),
        )

        assert isinstance(decision, ToolCallsDecision)
        assert len(decision.calls) == 1
        assert decision.calls[0].name == "my_tool"
        assert decision.calls[0].arguments == {"param": 1}
        assert decision.calls[0].id.startswith("call_")

    async def test_decide_next_step_handles_result_with_validation(
        self, mock_llm_client: AsyncMock
    ):
        result_payload = json.dumps(
            {
                "decision": "result",
                "result": {"name": "test", "value": 42},
            }
        )
        mock_llm_client.complete.return_value = _structured_response(result_payload)
        strategy = _make_strategy(mock_llm_client, stream=True)
        publisher = MockEventPublisher()

        decision = await strategy.decide_next_step(
            _function_info(return_type=MyOutput, instructions="do it"),
            [],
            _tool_registry(),
            publisher,
        )

        assert isinstance(decision, ResultDecision)
        assert isinstance(decision.result, MyOutput)
        assert decision.result.name == "test"
        assert decision.result.value == 42

        stream_callback_obj = mock_llm_client.complete.await_args.kwargs[
            "stream_callback"
        ]
        assert callable(stream_callback_obj)
        stream_callback = cast(Callable[[str], Awaitable[None]], stream_callback_obj)
        await stream_callback("tok")
        assert any(
            isinstance(event, LLMTokenReceived) and event.token == "tok"
            for event in publisher.events
        )

    async def test_decide_next_step_accepts_client_decoded_output(
        self, mock_llm_client: AsyncMock
    ) -> None:
        mock_llm_client.complete.return_value = LLMResponse(
            structured_output=LLMOutput.from_json(
                {"decision": "result", "result": "done"}
            )
        )
        strategy = _make_strategy(mock_llm_client)

        decision = await strategy.decide_next_step(
            _function_info(instructions="do it"),
            [],
            _tool_registry(),
            MockEventPublisher(),
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == "done"

    async def test_decide_next_step_publishes_reasoning_tokens(
        self, mock_llm_client: AsyncMock
    ):
        mock_llm_client.complete.return_value = _structured_response(
            '{"decision": "result", "result": "done"}'
        )
        strategy = _make_strategy(mock_llm_client, stream=True)
        publisher = MockEventPublisher()

        await strategy.decide_next_step(
            _function_info(instructions="do it"),
            [],
            _tool_registry(),
            publisher,
        )

        reasoning_callback_obj = mock_llm_client.complete.await_args.kwargs[
            "reasoning_callback"
        ]
        assert callable(reasoning_callback_obj)
        reasoning_callback = cast(
            Callable[[str], Awaitable[None]], reasoning_callback_obj
        )
        await reasoning_callback("thinking")
        assert any(
            isinstance(event, LLMReasoningTokenReceived) and event.token == "thinking"
            for event in publisher.events
        )

    async def test_decide_next_step_does_not_set_reasoning_callback_by_default(
        self, mock_llm_client: AsyncMock
    ):
        mock_llm_client.complete.return_value = _structured_response(
            '{"decision": "result", "result": "done"}'
        )
        strategy = _make_strategy(mock_llm_client)

        await strategy.decide_next_step(
            _function_info(instructions="do it"),
            [],
            _tool_registry(),
            MockEventPublisher(),
        )

        assert mock_llm_client.complete.await_args.kwargs["reasoning_callback"] is None

    async def test_decide_next_step_does_not_set_stream_callback_by_default(
        self, mock_llm_client: AsyncMock
    ):
        mock_llm_client.complete.return_value = _structured_response(
            '{"decision": "result", "result": "done"}'
        )
        strategy = _make_strategy(mock_llm_client)

        decision = await strategy.decide_next_step(
            _function_info(instructions="do it"),
            [],
            _tool_registry(),
            MockEventPublisher(),
        )

        assert isinstance(decision, ResultDecision)
        assert mock_llm_client.complete.await_args.kwargs["stream_callback"] is None

    async def test_decide_next_step_raises_on_validation_error(
        self, mock_llm_client: AsyncMock
    ):
        # result is missing required 'value' field — Pydantic should reject it
        mock_llm_client.complete.return_value = _structured_response(
            '{"decision": "result", "result": {"name": "test"}}'
        )
        strategy = _make_strategy(mock_llm_client)

        with pytest.raises(
            InvalidInferenceResponseError, match="LLM output failed validation"
        ):
            await strategy.decide_next_step(
                _function_info(return_type=MyOutput, instructions="do it"),
                [],
                _tool_registry(),
                MockEventPublisher(),
            )

    async def test_decide_next_step_handles_plain_string_output(
        self, mock_llm_client: AsyncMock
    ):
        mock_llm_client.complete.return_value = _structured_response(
            '{"decision": "result", "result": "Hello, world!"}'
        )
        strategy = _make_strategy(mock_llm_client)

        decision = await strategy.decide_next_step(
            _function_info(instructions="do it"),
            [],
            _tool_registry(),
            MockEventPublisher(),
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == "Hello, world!"

    async def test_decide_next_step_raises_when_result_null(
        self, mock_llm_client: AsyncMock
    ):
        mock_llm_client.complete.return_value = _structured_response(
            '{"decision": "result", "result": null}'
        )
        strategy = _make_strategy(mock_llm_client)

        with pytest.raises(
            InvalidInferenceResponseError, match="LLM output failed validation"
        ):
            await strategy.decide_next_step(
                _function_info(instructions="do it"),
                [],
                _tool_registry(),
                MockEventPublisher(),
            )


class TestResponseRepair:
    """Feedback-aware retry of invalid LLM responses (issue #35)."""

    VALID_RESULT = '{"decision": "result", "result": "done"}'

    def test_rejects_negative_budget(self):
        with pytest.raises(ValueError, match="non-negative"):
            _make_strategy(max_repair_attempts=-1)

    async def test_repairs_empty_response(self):
        client = AsyncMock()
        client.complete.side_effect = [
            LLMResponse(content=""),
            _structured_response(self.VALID_RESULT),
        ]
        strategy = _make_strategy(client)
        publisher = MockEventPublisher()

        decision = await strategy.decide_next_step(
            _function_info(), [], _tool_registry(), publisher
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == "done"
        assert client.complete.await_count == 2

        retry_messages = client.complete.await_args_list[1].kwargs["messages"]
        feedback = retry_messages[-1]
        assert feedback.role == "user"
        assert "The previous response was empty." in str(feedback.content)
        assert "Reason:" in str(feedback.content)

        assert any(
            isinstance(event, LLMResponseRepairAttempt) and event.attempt == 1
            for event in publisher.events
        )

    async def test_repairs_none_content(self):
        client = AsyncMock()
        client.complete.side_effect = [
            LLMResponse(content=None),
            _structured_response(self.VALID_RESULT),
        ]
        strategy = _make_strategy(client)

        decision = await strategy.decide_next_step(
            _function_info(), [], _tool_registry(), MockEventPublisher()
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == "done"

        retry_messages = client.complete.await_args_list[1].kwargs["messages"]
        assert "did not return structured output" in str(retry_messages[-1].content)

    async def test_repairs_client_response_decoding_error(self) -> None:
        partial = LLMResponse(content="malformed provider response")
        client = AsyncMock()
        client.complete.side_effect = [
            LLMResponseDecodingError(partial, "response could not be decoded"),
            _structured_response(self.VALID_RESULT),
        ]
        strategy = _make_strategy(client)

        decision = await strategy.decide_next_step(
            _function_info(), [], _tool_registry(), MockEventPublisher()
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == "done"
        retry_prompt = client.complete.await_args_list[1].kwargs["messages"][0]
        assert "malformed provider response" in str(retry_prompt.content)
        assert "response could not be decoded" in str(retry_prompt.content)

    async def test_repairs_schema_violation_and_echoes_invalid_output(self):
        invalid = '{"decision": "result", "result": {"name": "test"}}'
        valid = '{"decision": "result", "result": {"name": "test", "value": 42}}'
        client = AsyncMock()
        client.complete.side_effect = [
            _structured_response(invalid),
            _structured_response(valid),
        ]
        strategy = _make_strategy(client)

        decision = await strategy.decide_next_step(
            _function_info(return_type=MyOutput),
            [],
            _tool_registry(),
            MockEventPublisher(),
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == MyOutput(name="test", value=42)

        retry_prompt = client.complete.await_args_list[1].kwargs["messages"][0]
        assert retry_prompt.role == "user"
        assert invalid in str(retry_prompt.content)
        assert "Reason:" in str(retry_prompt.content)

    async def test_repairs_unknown_tool_call(self):
        invalid = json.dumps(
            {
                "decision": "tool_calls",
                "tool_calls": [{"name": "no_such_tool", "arguments": {}}],
            }
        )
        valid = json.dumps(
            {
                "decision": "tool_calls",
                "tool_calls": [{"name": "my_tool", "arguments": {"param": 1}}],
            }
        )
        client = AsyncMock()
        client.complete.side_effect = [
            _structured_response(invalid),
            _structured_response(valid),
        ]
        strategy = _make_strategy(client)

        decision = await strategy.decide_next_step(
            _function_info(), [], _tool_registry(my_tool), MockEventPublisher()
        )

        assert isinstance(decision, ToolCallsDecision)
        assert decision.calls[0].name == "my_tool"
        assert client.complete.await_count == 2

    async def test_repair_does_not_mutate_history(self):
        client = AsyncMock()
        client.complete.side_effect = [
            LLMResponse(content="not json"),
            _structured_response(self.VALID_RESULT),
        ]
        strategy = _make_strategy(client)
        history = [
            ToolCallsDecision(
                calls=[ToolCallRequest(id="1", name="search", arguments={"q": "x"})]
            ),
            ToolCallResult(tool_call_id="1", result="found"),
        ]
        snapshot = list(history)

        await strategy.decide_next_step(
            _function_info(), history, _tool_registry(), MockEventPublisher()
        )

        # Repair context is rendered into a new prompt without changing the
        # executor-owned decision history.
        assert history == snapshot
        first_messages = client.complete.await_args_list[0].kwargs["messages"]
        retry_messages = client.complete.await_args_list[1].kwargs["messages"]
        assert len(first_messages) == len(retry_messages) == 1
        assert first_messages[0].content != retry_messages[0].content

    async def test_raises_original_error_after_budget_exhausted(self):
        invalid = "not json"
        client = AsyncMock()
        client.complete.return_value = LLMResponse(content=invalid)
        strategy = _make_strategy(client, max_repair_attempts=2)

        with pytest.raises(
            InvalidInferenceResponseError, match="LLM output could not be decoded"
        ) as exc_info:
            await strategy.decide_next_step(
                _function_info(), [], _tool_registry(), MockEventPublisher()
            )

        assert client.complete.await_count == 3
        assert exc_info.value.raw_content == invalid
        assert exc_info.value.detail.startswith("LLM output could not be decoded")

    async def test_zero_budget_disables_repair(self):
        client = AsyncMock()
        client.complete.return_value = LLMResponse(content="not json")
        strategy = _make_strategy(client, max_repair_attempts=0)

        with pytest.raises(InvalidInferenceResponseError):
            await strategy.decide_next_step(
                _function_info(), [], _tool_registry(), MockEventPublisher()
            )

        assert client.complete.await_count == 1
