import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Never, cast
from unittest.mock import AsyncMock, Mock

import pytest

from sefia._tool_system import SignatureToolEntry, ToolEntry, ToolRegistry
from sefia.event_system import EventPublisher
from sefia.exceptions import InvalidInferenceResponseError
from sefia.inference import (
    FunctionInfo,
    ResultDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)
from sefia.llm import LLMInferenceStrategy, LLMResponse
from sefia.llm._strategy import (
    _OutputOnlyDirector,
    _ToolEnabledDirector,
    _ToolOnlyDirector,
)
from sefia.llm.events import (
    LLMReasoningTokenReceived,
    LLMResponseRepairAttempt,
    LLMTokenReceived,
)
from sefia.pydantic import PydanticModelBackend
from sefia.pydantic._json_utils import pydantic_json_default


class MockEventPublisher(EventPublisher):
    def __init__(self):
        super().__init__(handlers=[])
        self.events = []

    async def publish(self, event):
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


def _tool(func: Callable[..., Any]) -> ToolEntry:
    name = _BACKEND.tool_name(func)
    return SignatureToolEntry(
        func,
        name=name,
        schema_source=func,
        inspector=_BACKEND,
    )


def _tool_registry(*funcs: Callable[..., Any]) -> ToolRegistry:
    registry = ToolRegistry()
    for func in funcs:
        registry.add(func, name=_BACKEND.tool_name(func))
    return registry


@pytest.fixture
def mock_llm_client():
    return AsyncMock()


DUMMY_SCHEMA: dict = {}


def _resolve(schema: dict, root: dict) -> dict:
    if "$ref" in schema:
        key = schema["$ref"].split("/")[-1]
        return root["$defs"][key]
    return schema


def _decision_branch(schema: dict, decision: str) -> dict:
    if schema.get("properties", {}).get("decision", {}).get("const") == decision:
        return schema

    for candidate in schema["oneOf"]:
        branch = _resolve(candidate, schema)
        if branch["properties"]["decision"]["const"] == decision:
            return branch

    raise AssertionError(f"Decision branch not found: {decision}")


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
    def _strategy(self, llm_client, stream: bool = False):
        mock_formatter = Mock()
        mock_formatter.format_arguments.return_value = "<arguments/>"
        return LLMInferenceStrategy(
            llm_client=llm_client,
            decision_builder=PydanticModelBackend(),
            prompt_formatter=mock_formatter,
            json_default=pydantic_json_default,
            stream=stream,
        )

    def test_build_messages_correctly(self):
        strategy = self._strategy(AsyncMock())
        history = [
            ToolCallDecision(
                calls=[
                    ToolCallRequest(
                        id="1",
                        name="search",
                        arguments={"q": "日本語の検索クエリ"},
                    )
                ]
            ),
            ToolCallResult(tool_call_id="1", result="見つかりました"),
        ]

        director = strategy._create_director(str, [_tool(search)])
        messages = strategy._build_messages(
            _function_info(arguments={"arg": "val"}),
            history,
            DUMMY_SCHEMA,
            director,
        )

        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        assert messages[3].role == "tool"
        tool_calls = messages[2].tool_calls
        assert tool_calls is not None
        tool_arguments = tool_calls[0]["function"]["arguments"]
        assert "日本語の検索クエリ" in tool_arguments
        assert "\\u65e5" not in tool_arguments
        assert json.loads(tool_arguments) == {"q": "日本語の検索クエリ"}
        assert "見つかりました" in str(messages[3].content)
        assert "\\u898b" not in str(messages[3].content)
        assert json.loads(str(messages[3].content)) == "見つかりました"

    def test_build_decision_schema_hoists_nested_definitions(self):
        strategy = self._strategy(AsyncMock())

        director = strategy._create_director(list[MyIssue], [_tool(search)])
        schema = director.build_decision_schema()

        assert "MyIssue" in schema["$defs"]
        result_branch = _decision_branch(schema, "result")
        result_schema = result_branch["properties"]["result"]
        assert "$defs" not in result_schema
        assert result_schema["items"]["$ref"] == "#/$defs/MyIssue"
        # tool_calls items reference a per-tool call model in $defs, not an inline
        # blob, so each tool's arguments stay constrained by its own schema.
        tool_calls_branch = _decision_branch(schema, "tool_calls")
        tool_calls_array = tool_calls_branch["properties"]["tool_calls"]
        tool_call_ref = tool_calls_array["items"]["$ref"].split("/")[-1]
        assert tool_call_ref in schema["$defs"]
        assert tool_call_ref.endswith("ToolCall")

    def test_build_decision_schema_requires_non_null_result_without_tools(self):
        strategy = self._strategy(AsyncMock())

        director = strategy._create_director(list[MyIssue], [])
        schema = director.build_decision_schema()

        assert schema["required"] == ["decision", "result"]
        assert schema["properties"]["decision"]["const"] == "result"
        result_schema = schema["properties"]["result"]
        assert result_schema["type"] == "array"
        assert "oneOf" not in result_schema

    def test_build_decision_schema_uses_decision_discriminator_with_tools(self):
        strategy = self._strategy(AsyncMock())

        director = strategy._create_director(list[MyIssue], [_tool(search)])
        schema = director.build_decision_schema()

        assert schema["discriminator"]["propertyName"] == "decision"
        assert set(schema["discriminator"]["mapping"]) == {
            "tool_calls",
            "result",
        }
        assert _decision_branch(schema, "tool_calls")["required"] == [
            "decision",
            "tool_calls",
        ]
        assert _decision_branch(schema, "result")["required"] == [
            "decision",
            "result",
        ]

    def test_build_messages_tells_no_tool_agent_to_return_empty_collection(self):
        strategy = self._strategy(AsyncMock())

        director = strategy._create_director(list[MyIssue], [])
        messages = strategy._build_messages(
            _function_info(return_type=list[MyIssue]),
            [],
            DUMMY_SCHEMA,
            director,
        )

        assert "empty collection instead of null" in str(messages[0].content)

    async def test_decide_next_step_handles_tool_calls(self, mock_llm_client):
        tool_calls_payload = json.dumps(
            {
                "decision": "tool_calls",
                "tool_calls": [{"name": "my_tool", "arguments": {"param": 1}}],
            }
        )
        mock_llm_client.complete.return_value = LLMResponse(content=tool_calls_payload)
        strategy = self._strategy(mock_llm_client, stream=True)

        decision = await strategy.decide_next_step(
            _function_info(instructions="do it"),
            [],
            _tool_registry(my_tool),
            MockEventPublisher(),
        )

        assert isinstance(decision, ToolCallDecision)
        assert len(decision.calls) == 1
        assert decision.calls[0].name == "my_tool"
        assert decision.calls[0].arguments == {"param": 1}
        assert decision.calls[0].id.startswith("call_")

    async def test_decide_next_step_handles_result_with_validation(
        self, mock_llm_client
    ):
        result_payload = json.dumps(
            {
                "decision": "result",
                "result": {"name": "test", "value": 42},
            }
        )
        mock_llm_client.complete.return_value = LLMResponse(content=result_payload)
        strategy = self._strategy(mock_llm_client, stream=True)
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

    async def test_decide_next_step_publishes_reasoning_tokens(self, mock_llm_client):
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"decision": "result", "result": "done"}'
        )
        strategy = self._strategy(mock_llm_client, stream=True)
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
        self, mock_llm_client
    ):
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"decision": "result", "result": "done"}'
        )
        strategy = self._strategy(mock_llm_client)

        await strategy.decide_next_step(
            _function_info(instructions="do it"),
            [],
            _tool_registry(),
            MockEventPublisher(),
        )

        assert mock_llm_client.complete.await_args.kwargs["reasoning_callback"] is None

    async def test_decide_next_step_does_not_set_stream_callback_by_default(
        self, mock_llm_client
    ):
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"decision": "result", "result": "done"}'
        )
        strategy = self._strategy(mock_llm_client)

        decision = await strategy.decide_next_step(
            _function_info(instructions="do it"),
            [],
            _tool_registry(),
            MockEventPublisher(),
        )

        assert isinstance(decision, ResultDecision)
        assert mock_llm_client.complete.await_args.kwargs["stream_callback"] is None

    async def test_decide_next_step_raises_on_validation_error(self, mock_llm_client):
        # result is missing required 'value' field — Pydantic should reject it
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"decision": "result", "result": {"name": "test"}}'
        )
        strategy = self._strategy(mock_llm_client)

        with pytest.raises(
            InvalidInferenceResponseError, match="LLM output failed validation"
        ):
            await strategy.decide_next_step(
                _function_info(return_type=MyOutput, instructions="do it"),
                [],
                _tool_registry(),
                MockEventPublisher(),
            )

    async def test_decide_next_step_handles_plain_string_output(self, mock_llm_client):
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"decision": "result", "result": "Hello, world!"}'
        )
        strategy = self._strategy(mock_llm_client)

        decision = await strategy.decide_next_step(
            _function_info(instructions="do it"),
            [],
            _tool_registry(),
            MockEventPublisher(),
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == "Hello, world!"

    async def test_decide_next_step_raises_when_result_null(self, mock_llm_client):
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"decision": "result", "result": null}'
        )
        strategy = self._strategy(mock_llm_client)

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

    def _strategy(self, llm_client, max_repair_attempts: int = 2):
        mock_formatter = Mock()
        mock_formatter.format_arguments.return_value = "<arguments/>"
        return LLMInferenceStrategy(
            llm_client=llm_client,
            decision_builder=PydanticModelBackend(),
            prompt_formatter=mock_formatter,
            json_default=pydantic_json_default,
            max_repair_attempts=max_repair_attempts,
        )

    def test_rejects_negative_budget(self):
        with pytest.raises(ValueError, match="non-negative"):
            self._strategy(AsyncMock(), max_repair_attempts=-1)

    async def test_repairs_empty_response(self):
        client = AsyncMock()
        client.complete.side_effect = [
            LLMResponse(content=""),
            LLMResponse(content=self.VALID_RESULT),
        ]
        strategy = self._strategy(client)
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
        assert "invalid" in str(feedback.content)
        assert "Your previous response was empty." in str(feedback.content)
        assert "Error:" in str(feedback.content)

        assert any(
            isinstance(event, LLMResponseRepairAttempt) and event.attempt == 1
            for event in publisher.events
        )

    async def test_repairs_none_content(self):
        client = AsyncMock()
        client.complete.side_effect = [
            LLMResponse(content=None),
            LLMResponse(content=self.VALID_RESULT),
        ]
        strategy = self._strategy(client)

        decision = await strategy.decide_next_step(
            _function_info(), [], _tool_registry(), MockEventPublisher()
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == "done"

        retry_messages = client.complete.await_args_list[1].kwargs["messages"]
        assert "did not provide a response content" in str(retry_messages[-1].content)

    async def test_repairs_schema_violation_and_echoes_invalid_output(self):
        invalid = '{"decision": "result", "result": {"name": "test"}}'
        valid = '{"decision": "result", "result": {"name": "test", "value": 42}}'
        client = AsyncMock()
        client.complete.side_effect = [
            LLMResponse(content=invalid),
            LLMResponse(content=valid),
        ]
        strategy = self._strategy(client)

        decision = await strategy.decide_next_step(
            _function_info(return_type=MyOutput),
            [],
            _tool_registry(),
            MockEventPublisher(),
        )

        assert isinstance(decision, ResultDecision)
        assert decision.result == MyOutput(name="test", value=42)

        retry_messages = client.complete.await_args_list[1].kwargs["messages"]
        # The invalid output is echoed back as the assistant turn, followed by
        # the corrective user message.
        assert retry_messages[-2].role == "assistant"
        assert retry_messages[-2].content == invalid
        assert retry_messages[-1].role == "user"
        assert "Error:" in str(retry_messages[-1].content)

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
            LLMResponse(content=invalid),
            LLMResponse(content=valid),
        ]
        strategy = self._strategy(client)

        decision = await strategy.decide_next_step(
            _function_info(), [], _tool_registry(my_tool), MockEventPublisher()
        )

        assert isinstance(decision, ToolCallDecision)
        assert decision.calls[0].name == "my_tool"
        assert client.complete.await_count == 2

    async def test_repair_does_not_mutate_history(self):
        client = AsyncMock()
        client.complete.side_effect = [
            LLMResponse(content="not json"),
            LLMResponse(content=self.VALID_RESULT),
        ]
        strategy = self._strategy(client)
        history = [
            ToolCallDecision(
                calls=[ToolCallRequest(id="1", name="search", arguments={"q": "x"})]
            ),
            ToolCallResult(tool_call_id="1", result="found"),
        ]
        snapshot = list(history)

        await strategy.decide_next_step(
            _function_info(), history, _tool_registry(), MockEventPublisher()
        )

        # The repair exchange lives only in the per-attempt messages; the step
        # history the executor owns is untouched.
        assert history == snapshot
        first_messages = client.complete.await_args_list[0].kwargs["messages"]
        retry_messages = client.complete.await_args_list[1].kwargs["messages"]
        assert len(retry_messages) == len(first_messages) + 2

    async def test_raises_original_error_after_budget_exhausted(self):
        invalid = "not json"
        client = AsyncMock()
        client.complete.return_value = LLMResponse(content=invalid)
        strategy = self._strategy(client, max_repair_attempts=2)

        with pytest.raises(
            InvalidInferenceResponseError, match="LLM output failed validation"
        ) as exc_info:
            await strategy.decide_next_step(
                _function_info(), [], _tool_registry(), MockEventPublisher()
            )

        assert client.complete.await_count == 3
        assert exc_info.value.raw_content == invalid
        assert exc_info.value.detail.startswith("LLM output failed validation")

    async def test_zero_budget_disables_repair(self):
        client = AsyncMock()
        client.complete.return_value = LLMResponse(content="not json")
        strategy = self._strategy(client, max_repair_attempts=0)

        with pytest.raises(InvalidInferenceResponseError):
            await strategy.decide_next_step(
                _function_info(), [], _tool_registry(), MockEventPublisher()
            )

        assert client.complete.await_count == 1


class TestToolOnlyDirector:
    """Tests for _ToolOnlyDirector — the Never return type mode."""

    def _strategy(self):
        mock_formatter = Mock()
        mock_formatter.format_arguments.return_value = "<arguments/>"
        return LLMInferenceStrategy(
            llm_client=AsyncMock(),
            decision_builder=PydanticModelBackend(),
            prompt_formatter=mock_formatter,
            json_default=pydantic_json_default,
        )

    def test_create_director_returns_tool_only_for_never(self):
        strategy = self._strategy()
        director = strategy._create_director(Never, [_tool(chat_tool)])
        assert isinstance(director, _ToolOnlyDirector)

    def test_create_director_raises_for_never_without_tools(self):
        strategy = self._strategy()
        with pytest.raises(ValueError, match="must have tools available"):
            strategy._create_director(Never, [])

    def test_build_decision_schema_has_no_result_field(self):
        strategy = self._strategy()
        director = strategy._create_director(Never, [_tool(chat_tool)])
        schema = director.build_decision_schema()

        assert "result" not in schema.get("properties", {})
        assert schema["properties"]["decision"]["const"] == "tool_calls"
        assert "tool_calls" in schema["properties"]
        assert schema["required"] == ["decision", "tool_calls"]

    def test_build_system_prompt_instructs_tool_only(self):
        strategy = self._strategy()
        director = strategy._create_director(Never, [_tool(chat_tool)])
        schema = director.build_decision_schema()
        prompt = director.build_system_prompt_addition(schema)

        assert "result" not in prompt or "There is no `result`" in prompt
        assert "tool_calls" in prompt
        assert "### Response Format" in prompt
        assert '"$defs"' not in prompt

    def test_process_decision_accepts_tool_calls(self):
        strategy = self._strategy()
        director = strategy._create_director(Never, [_tool(chat_tool)])

        result = director.process_response_data(
            {
                "decision": "tool_calls",
                "tool_calls": [{"name": "chat_tool", "arguments": {}}],
            }
        )

        assert isinstance(result, ToolCallDecision)
        assert result.calls[0].name == "chat_tool"

    async def test_decide_next_step_returns_tool_call_decision_for_never(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = LLMResponse(
            content=(
                '{"decision": "tool_calls", '
                '"tool_calls": [{"name": "chat_tool", "arguments": {}}]}'
            )
        )
        mock_formatter = Mock()
        mock_formatter.format_arguments.return_value = "<arguments/>"
        strategy = LLMInferenceStrategy(
            llm_client=mock_client,
            decision_builder=PydanticModelBackend(),
            prompt_formatter=mock_formatter,
        )

        result = await strategy.decide_next_step(
            _function_info(return_type=Never, instructions="chat"),
            [],
            _tool_registry(chat_tool),
            MockEventPublisher(),
        )

        assert isinstance(result, ToolCallDecision)

    async def test_decide_next_step_raises_when_llm_returns_result_for_never(
        self,
    ):
        mock_client = AsyncMock()
        mock_client.complete.return_value = LLMResponse(
            content='{"decision": "result", "result": "bye"}'
        )
        mock_formatter = Mock()
        mock_formatter.format_arguments.return_value = "<arguments/>"
        strategy = LLMInferenceStrategy(
            llm_client=mock_client,
            decision_builder=PydanticModelBackend(),
            prompt_formatter=mock_formatter,
        )

        with pytest.raises(
            InvalidInferenceResponseError, match="LLM output failed validation"
        ):
            await strategy.decide_next_step(
                _function_info(return_type=Never, instructions="chat"),
                [],
                _tool_registry(chat_tool),
                MockEventPublisher(),
            )


class TestToolEnabledDirector:
    """Tests for _ToolEnabledDirector — tools and result are both allowed."""

    def _director(self, output_type: Any = str):
        return _ToolEnabledDirector(
            PydanticModelBackend(), output_type, [_tool(search)]
        )

    def test_build_decision_schema_has_decision_branches(self):
        schema = self._director().build_decision_schema()

        assert schema["discriminator"]["propertyName"] == "decision"
        assert _decision_branch(schema, "tool_calls")["required"] == [
            "decision",
            "tool_calls",
        ]
        assert _decision_branch(schema, "result")["required"] == [
            "decision",
            "result",
        ]

    def test_build_system_prompt_mentions_both_options(self):
        director = self._director()
        prompt = director.build_system_prompt_addition(director.build_decision_schema())

        assert "tool_calls" in prompt
        assert "result" in prompt

    def test_process_decision_returns_tool_call_decision(self):
        director = self._director()
        result = director.process_response_data(
            {
                "decision": "tool_calls",
                "tool_calls": [{"name": "search", "arguments": {"q": "x"}}],
            },
        )

        assert isinstance(result, ToolCallDecision)
        assert result.calls[0].name == "search"
        assert result.calls[0].arguments == {"q": "x"}

    def test_process_decision_returns_result_decision(self):
        director = self._director(output_type=str)
        result = director.process_response_data(
            {"decision": "result", "result": "done"}
        )

        assert isinstance(result, ResultDecision)
        assert result.result == "done"

    def test_process_decision_validates_result_type(self):
        director = self._director(output_type=MyOutput)
        result = director.process_response_data(
            {
                "decision": "result",
                "result": {"name": "ok", "value": 7},
            }
        )

        assert isinstance(result, ResultDecision)
        assert isinstance(result.result, MyOutput)
        assert result.result.value == 7

    def test_process_decision_raises_when_decision_branch_is_incomplete(self):
        director = self._director()
        with pytest.raises(
            ValueError,
            match="Decision validation failed",
        ):
            director.process_response_data({"decision": "tool_calls"})

    def test_decision_validation_rejects_fields_from_other_branch(self):
        director = self._director()
        with pytest.raises(ValueError, match="Decision validation failed"):
            director.process_response_data(
                {
                    "decision": "tool_calls",
                    "result": "extra",
                    "tool_calls": [{"name": "search", "arguments": {"q": "x"}}],
                },
            )


class TestOutputOnlyDirector:
    """Tests for _OutputOnlyDirector — no tools, result required."""

    def _director(self, output_type: Any = str):
        return _OutputOnlyDirector(PydanticModelBackend(), output_type, [])

    def test_build_decision_schema_has_only_result(self):
        schema = self._director().build_decision_schema()

        assert "tool_calls" not in schema.get("properties", {})
        assert "result" in schema["properties"]
        assert schema["properties"]["decision"]["const"] == "result"
        assert schema["required"] == ["decision", "result"]

    def test_build_system_prompt_mentions_no_tools(self):
        director = self._director()
        prompt = director.build_system_prompt_addition(director.build_decision_schema())

        assert "No tools are available" in prompt

    def test_process_decision_returns_result(self):
        director = self._director(output_type=str)
        result = director.process_response_data(
            {"decision": "result", "result": "hello"}
        )

        assert isinstance(result, ResultDecision)
        assert result.result == "hello"

    def test_process_decision_validates_structured_output(self):
        director = self._director(output_type=MyOutput)
        result = director.process_response_data(
            {
                "decision": "result",
                "result": {"name": "test", "value": 99},
            }
        )

        assert isinstance(result, ResultDecision)
        assert isinstance(result.result, MyOutput)
        assert result.result.name == "test"

    def test_decision_model_rejects_null_result(self):
        director = self._director()
        with pytest.raises(ValueError, match="Decision validation failed"):
            director.process_response_data({"decision": "result", "result": None})
