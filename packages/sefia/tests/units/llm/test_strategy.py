import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Never, cast
from unittest.mock import AsyncMock, Mock

import pytest

from sefia._tool_system import ToolRegistry
from sefia.event_system import Event, EventPublisher
from sefia.exceptions import InvalidInferenceResponseError
from sefia.inference import (
    FunctionInfo,
    ResultDecision,
    ToolCallsDecision,
    ToolCallResult,
)
from sefia.llm import (
    DecisionPrompt,
    LLMCompletion,
    LLMInferenceStrategy,
    MarkdownPromptRenderer,
    PromptRenderer,
    ToolCall,
)
from sefia.llm.exceptions import LLMCompletionDecodingError
from sefia.llm.events import (
    LLMReasoningTokenReceived,
    DecisionRepairAttempt,
    LLMTokenReceived,
)
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import NativeDecisionTransport, StructuredDecisionTransport
from sefia.pydantic import PydanticModelBackend
from sefia.pydantic._json_utils import pydantic_json_default
from sefia.testing import make_function_info, make_tool_call_request


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
    llm_client: Any = None,
    *,
    stream: bool = False,
    max_repair_attempts: int = 2,
    prompt_renderer: PromptRenderer | None = None,
) -> LLMInferenceStrategy:
    """The strategy under test, with a stub prompt renderer."""
    client = llm_client if llm_client is not None else AsyncMock()
    return LLMInferenceStrategy(
        llm_client=client,
        result_format_factory=PydanticModelBackend(),
        prompt_renderer=(
            prompt_renderer
            if prompt_renderer is not None
            else MarkdownPromptRenderer(json_default=pydantic_json_default)
        ),
        decision_transport=StructuredDecisionTransport(),
        stream=stream,
        max_repair_attempts=max_repair_attempts,
    )


def _recording_renderer() -> Mock:
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "prompt"
    renderer.render_tool_result.return_value = "tool result"
    return renderer


def _retry_prompt(renderer: Mock) -> DecisionPrompt:
    return cast(DecisionPrompt, renderer.render.call_args_list[1].args[0])


def _structured_completion(content: str) -> LLMCompletion:
    return LLMCompletion(
        content=content,
        structured_output=StructuredData.parse_json(content),
    )


def _function_info(
    return_type: Any = str,
    arguments: dict[str, Any] | None = None,
    type_hints: dict[str, Any] | None = None,
    instructions: str = "instructions",
) -> FunctionInfo:
    return make_function_info(
        instructions=instructions,
        bound_arguments=arguments,
        type_hints=type_hints,
        return_type=return_type,
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
        mock_llm_client.complete.return_value = _structured_completion(
            tool_calls_payload
        )
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

    async def test_never_return_type_accepts_tool_call_decision(self) -> None:
        client = AsyncMock()
        content = (
            '{"decision": "tool_calls", '
            '"tool_calls": [{"name": "chat_tool", "arguments": {}}]}'
        )
        client.complete.return_value = _structured_completion(content)

        decision = await _make_strategy(client).decide_next_step(
            _function_info(return_type=Never, instructions="chat"),
            [],
            _tool_registry(chat_tool),
            MockEventPublisher(),
        )

        assert isinstance(decision, ToolCallsDecision)
        assert decision.calls[0].name == "chat_tool"

    async def test_never_return_type_rejects_result_decision(self) -> None:
        client = AsyncMock()
        client.complete.return_value = _structured_completion(
            '{"decision": "result", "result": "bye"}'
        )

        with pytest.raises(
            InvalidInferenceResponseError, match="LLM decision failed validation"
        ):
            await _make_strategy(client, max_repair_attempts=0).decide_next_step(
                _function_info(return_type=Never, instructions="chat"),
                [],
                _tool_registry(chat_tool),
                MockEventPublisher(),
            )

    async def test_decide_next_step_handles_result_with_validation(
        self, mock_llm_client: AsyncMock
    ):
        result_payload = json.dumps(
            {
                "decision": "result",
                "result": {"name": "test", "value": 42},
            }
        )
        mock_llm_client.complete.return_value = _structured_completion(result_payload)
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
        mock_llm_client.complete.return_value = LLMCompletion(
            structured_output=StructuredData.from_json(
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
        mock_llm_client.complete.return_value = _structured_completion(
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
        mock_llm_client.complete.return_value = _structured_completion(
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
        mock_llm_client.complete.return_value = _structured_completion(
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
        mock_llm_client.complete.return_value = _structured_completion(
            '{"decision": "result", "result": {"name": "test"}}'
        )
        strategy = _make_strategy(mock_llm_client)

        with pytest.raises(
            InvalidInferenceResponseError, match="LLM decision failed validation"
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
        mock_llm_client.complete.return_value = _structured_completion(
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
        mock_llm_client.complete.return_value = _structured_completion(
            '{"decision": "result", "result": null}'
        )
        strategy = _make_strategy(mock_llm_client)

        with pytest.raises(
            InvalidInferenceResponseError, match="LLM decision failed validation"
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
            LLMCompletion(content=""),
            _structured_completion(self.VALID_RESULT),
        ]
        renderer = _recording_renderer()
        strategy = _make_strategy(
            client, prompt_renderer=cast(PromptRenderer, renderer)
        )
        publisher = MockEventPublisher()

        decision = await strategy.decide_next_step(
            _function_info(), [], _tool_registry(), publisher
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == "done"
        assert client.complete.await_count == 2

        rejected = _retry_prompt(renderer).rejected
        assert rejected is not None
        assert rejected.content == ""
        assert rejected.reason

        assert any(
            isinstance(event, DecisionRepairAttempt) and event.attempt == 1
            for event in publisher.events
        )

    async def test_repairs_none_content(self):
        client = AsyncMock()
        client.complete.side_effect = [
            LLMCompletion(content=None),
            _structured_completion(self.VALID_RESULT),
        ]
        renderer = _recording_renderer()
        strategy = _make_strategy(
            client, prompt_renderer=cast(PromptRenderer, renderer)
        )

        decision = await strategy.decide_next_step(
            _function_info(), [], _tool_registry(), MockEventPublisher()
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == "done"

        rejected = _retry_prompt(renderer).rejected
        assert rejected is not None
        assert rejected.content is None
        assert "did not return structured output" in rejected.reason

    async def test_repairs_client_completion_decoding_error(self) -> None:
        partial = LLMCompletion(content="malformed provider response")
        client = AsyncMock()
        client.complete.side_effect = [
            LLMCompletionDecodingError(partial, "response could not be decoded"),
            _structured_completion(self.VALID_RESULT),
        ]
        renderer = _recording_renderer()
        strategy = _make_strategy(
            client, prompt_renderer=cast(PromptRenderer, renderer)
        )

        decision = await strategy.decide_next_step(
            _function_info(), [], _tool_registry(), MockEventPublisher()
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == "done"
        rejected = _retry_prompt(renderer).rejected
        assert rejected is not None
        assert rejected.content == "malformed provider response"
        assert rejected.reason.endswith("response could not be decoded")

    async def test_repairs_schema_violation_and_echoes_invalid_output(self):
        invalid = '{"decision": "result", "result": {"name": "test"}}'
        valid = '{"decision": "result", "result": {"name": "test", "value": 42}}'
        client = AsyncMock()
        client.complete.side_effect = [
            _structured_completion(invalid),
            _structured_completion(valid),
        ]
        renderer = _recording_renderer()
        strategy = _make_strategy(
            client, prompt_renderer=cast(PromptRenderer, renderer)
        )

        decision = await strategy.decide_next_step(
            _function_info(return_type=MyOutput),
            [],
            _tool_registry(),
            MockEventPublisher(),
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == MyOutput(name="test", value=42)

        rejected = _retry_prompt(renderer).rejected
        assert rejected is not None
        assert json.loads(rejected.content or "") == json.loads(invalid)
        assert rejected.reason

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
            _structured_completion(invalid),
            _structured_completion(valid),
        ]
        strategy = _make_strategy(client)

        decision = await strategy.decide_next_step(
            _function_info(), [], _tool_registry(my_tool), MockEventPublisher()
        )

        assert isinstance(decision, ToolCallsDecision)
        assert decision.calls[0].name == "my_tool"
        assert client.complete.await_count == 2

    async def test_native_repair_includes_rejected_tool_call(self) -> None:
        client = AsyncMock()
        client.complete.side_effect = [
            LLMCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="no_such_tool",
                        arguments=StructuredData.from_json({"query": "lost"}),
                    )
                ]
            ),
            LLMCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="return_result",
                        arguments=StructuredData.from_json({"result": "done"}),
                    )
                ]
            ),
        ]
        renderer = _recording_renderer()
        strategy = LLMInferenceStrategy(
            llm_client=client,
            result_format_factory=PydanticModelBackend(),
            prompt_renderer=cast(PromptRenderer, renderer),
            decision_transport=NativeDecisionTransport(),
        )

        decision = await strategy.decide_next_step(
            _function_info(), [], _tool_registry(my_tool), MockEventPublisher()
        )

        assert isinstance(decision, ResultDecision)
        rejected = _retry_prompt(renderer).rejected
        assert rejected is not None
        assert json.loads(rejected.content or "") == {
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "no_such_tool",
                    "arguments": {"query": "lost"},
                }
            ]
        }

    async def test_repair_does_not_mutate_history(self):
        client = AsyncMock()
        client.complete.side_effect = [
            LLMCompletion(content="not json"),
            _structured_completion(self.VALID_RESULT),
        ]
        strategy = _make_strategy(client)
        history = [
            ToolCallsDecision(
                calls=[
                    make_tool_call_request(id="1", name="search", arguments={"q": "x"})
                ]
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
        client.complete.return_value = LLMCompletion(content=invalid)
        strategy = _make_strategy(client, max_repair_attempts=2)

        with pytest.raises(
            InvalidInferenceResponseError, match="LLM decision could not be decoded"
        ) as exc_info:
            await strategy.decide_next_step(
                _function_info(), [], _tool_registry(), MockEventPublisher()
            )

        assert client.complete.await_count == 3
        assert exc_info.value.raw_content == invalid
        assert exc_info.value.detail.startswith("LLM decision could not be decoded")

    async def test_zero_budget_disables_repair(self):
        client = AsyncMock()
        client.complete.return_value = LLMCompletion(content="not json")
        strategy = _make_strategy(client, max_repair_attempts=0)

        with pytest.raises(InvalidInferenceResponseError):
            await strategy.decide_next_step(
                _function_info(), [], _tool_registry(), MockEventPublisher()
            )

        assert client.complete.await_count == 1
