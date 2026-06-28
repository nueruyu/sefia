import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Never, cast
from unittest.mock import AsyncMock, Mock

import pytest

from sefia._tool_system import ToolRegistry
from sefia.event_system import EventPublisher
from sefia.exceptions import InvalidInferenceResponseError
from sefia.inference import (
    FinalAnswerDecision,
    FunctionInfo,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)
from sefia.llm import LLMInferenceStrategy, LLMResponse
from sefia.llm._strategy import (
    _OutputOnlyDirector,
    _ToolEnabledDirector,
    _ToolOnlyDirector,
    _ToolSpec,
)
from sefia.llm.events import LLMTokenReceived
from sefia.pydantic import PydanticModelInspector
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


_INSPECTOR = PydanticModelInspector()


def _spec(func: Callable[..., Any]) -> _ToolSpec:
    name = _INSPECTOR.get_function_name(func)
    return _ToolSpec(
        name=name,
        function=func,
        schema=_INSPECTOR.get_function_schema(func, name=name),
    )


def _tool_registry(*funcs: Callable[..., Any]) -> ToolRegistry:
    registry = ToolRegistry()
    for func in funcs:
        registry.add(func)
    return registry


@pytest.fixture
def mock_llm_client():
    return AsyncMock()


DUMMY_SCHEMA: dict = {}


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
            model_inspector=PydanticModelInspector(),
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

        director = strategy._create_director(str, [_spec(search)])
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

        director = strategy._create_director(list[MyIssue], [_spec(search)])
        schema = director.build_decision_schema()

        assert "MyIssue" in schema["$defs"]
        final_answer_schema = schema["properties"]["final_answer"]["anyOf"][0]
        assert "$defs" not in final_answer_schema
        assert final_answer_schema["items"]["$ref"] == "#/$defs/MyIssue"
        # tool_calls items reference a per-tool call model in $defs, not an inline
        # blob, so each tool's arguments stay constrained by its own schema.
        tool_calls_array = next(
            candidate
            for candidate in schema["properties"]["tool_calls"]["anyOf"]
            if candidate.get("type") == "array"
        )
        tool_call_ref = tool_calls_array["items"]["$ref"].split("/")[-1]
        assert tool_call_ref in schema["$defs"]
        assert tool_call_ref.endswith("ToolCall")

    def test_build_decision_schema_requires_non_null_answer_without_tools(self):
        strategy = self._strategy(AsyncMock())

        director = strategy._create_director(list[MyIssue], [])
        schema = director.build_decision_schema()

        assert schema["required"] == ["final_answer"]
        final_answer_schema = schema["properties"]["final_answer"]
        assert final_answer_schema["type"] == "array"
        assert "oneOf" not in final_answer_schema

    def test_build_decision_schema_requires_nullable_fields_with_tools(self):
        strategy = self._strategy(AsyncMock())

        director = strategy._create_director(list[MyIssue], [_spec(search)])
        schema = director.build_decision_schema()

        assert schema["required"] == ["final_answer", "tool_calls"]
        assert {"type": "null"} in schema["properties"]["final_answer"]["anyOf"]
        assert {"type": "null"} in schema["properties"]["tool_calls"]["anyOf"]

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
                "final_answer": None,
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

    async def test_decide_next_step_handles_final_answer_with_validation(
        self, mock_llm_client
    ):
        final_answer_payload = json.dumps(
            {"final_answer": {"name": "test", "value": 42}}
        )
        mock_llm_client.complete.return_value = LLMResponse(
            content=final_answer_payload
        )
        strategy = self._strategy(mock_llm_client, stream=True)
        publisher = MockEventPublisher()

        decision = await strategy.decide_next_step(
            _function_info(return_type=MyOutput, instructions="do it"),
            [],
            _tool_registry(),
            publisher,
        )

        assert isinstance(decision, FinalAnswerDecision)
        assert isinstance(decision.answer, MyOutput)
        assert decision.answer.name == "test"
        assert decision.answer.value == 42

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

    async def test_decide_next_step_does_not_set_stream_callback_by_default(
        self, mock_llm_client
    ):
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"final_answer": "done"}'
        )
        strategy = self._strategy(mock_llm_client)

        decision = await strategy.decide_next_step(
            _function_info(instructions="do it"),
            [],
            _tool_registry(),
            MockEventPublisher(),
        )

        assert isinstance(decision, FinalAnswerDecision)
        assert mock_llm_client.complete.await_args.kwargs["stream_callback"] is None

    async def test_decide_next_step_raises_on_validation_error(self, mock_llm_client):
        # final_answer is missing required 'value' field — Pydantic should reject it
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"final_answer": {"name": "test"}}'
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
            content='{"final_answer": "Hello, world!"}'
        )
        strategy = self._strategy(mock_llm_client)

        decision = await strategy.decide_next_step(
            _function_info(instructions="do it"),
            [],
            _tool_registry(),
            MockEventPublisher(),
        )

        assert isinstance(decision, FinalAnswerDecision)
        assert decision.answer == "Hello, world!"

    async def test_decide_next_step_raises_when_final_answer_null(
        self, mock_llm_client
    ):
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"final_answer": null}'
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


class TestToolOnlyDirector:
    """Tests for _ToolOnlyDirector — the Never return type mode."""

    def _strategy(self):
        mock_formatter = Mock()
        mock_formatter.format_arguments.return_value = "<arguments/>"
        return LLMInferenceStrategy(
            llm_client=AsyncMock(),
            model_inspector=PydanticModelInspector(),
            prompt_formatter=mock_formatter,
            json_default=pydantic_json_default,
        )

    def test_create_director_returns_tool_only_for_never(self):
        strategy = self._strategy()
        director = strategy._create_director(Never, [_spec(chat_tool)])
        assert isinstance(director, _ToolOnlyDirector)

    def test_create_director_raises_for_never_without_tools(self):
        strategy = self._strategy()
        with pytest.raises(ValueError, match="must have tools available"):
            strategy._create_director(Never, [])

    def test_build_decision_schema_has_no_final_answer_field(self):
        strategy = self._strategy()
        director = strategy._create_director(Never, [_spec(chat_tool)])
        schema = director.build_decision_schema()

        assert "final_answer" not in schema.get("properties", {})
        assert "tool_calls" in schema["properties"]
        assert schema["required"] == ["tool_calls"]

    def test_build_system_prompt_instructs_tool_only(self):
        strategy = self._strategy()
        director = strategy._create_director(Never, [_spec(chat_tool)])
        schema = director.build_decision_schema()
        prompt = director.build_system_prompt_addition(schema)

        assert "final_answer" not in prompt or "There is no `final_answer`" in prompt
        assert "tool_calls" in prompt

    def test_process_decision_accepts_tool_calls(self):
        strategy = self._strategy()
        director = strategy._create_director(Never, [_spec(chat_tool)])

        decision = director.decision_model.validate(
            {"tool_calls": [{"name": "chat_tool", "arguments": {}}]}
        )
        result = director.process_decision(decision)

        assert isinstance(result, ToolCallDecision)
        assert result.calls[0].name == "chat_tool"

    async def test_decide_next_step_returns_tool_call_decision_for_never(self):
        mock_client = AsyncMock()
        mock_client.complete.return_value = LLMResponse(
            content='{"tool_calls": [{"name": "chat_tool", "arguments": {}}]}'
        )
        mock_formatter = Mock()
        mock_formatter.format_arguments.return_value = "<arguments/>"
        strategy = LLMInferenceStrategy(
            llm_client=mock_client,
            model_inspector=PydanticModelInspector(),
            prompt_formatter=mock_formatter,
        )

        result = await strategy.decide_next_step(
            _function_info(return_type=Never, instructions="chat"),
            [],
            _tool_registry(chat_tool),
            MockEventPublisher(),
        )

        assert isinstance(result, ToolCallDecision)

    async def test_decide_next_step_raises_when_llm_returns_final_answer_for_never(
        self,
    ):
        mock_client = AsyncMock()
        mock_client.complete.return_value = LLMResponse(
            content='{"tool_calls": null, "final_answer": "bye"}'
        )
        mock_formatter = Mock()
        mock_formatter.format_arguments.return_value = "<arguments/>"
        strategy = LLMInferenceStrategy(
            llm_client=mock_client,
            model_inspector=PydanticModelInspector(),
            prompt_formatter=mock_formatter,
        )

        with pytest.raises(
            InvalidInferenceResponseError, match="must contain 'tool_calls'"
        ):
            await strategy.decide_next_step(
                _function_info(return_type=Never, instructions="chat"),
                [],
                _tool_registry(chat_tool),
                MockEventPublisher(),
            )


class TestToolEnabledDirector:
    """Tests for _ToolEnabledDirector — tools available, final answer also allowed."""

    def _director(self, output_type: Any = str):
        return _ToolEnabledDirector(
            PydanticModelInspector(), output_type, [_spec(search)]
        )

    def _decision(self, director, data):
        return director.decision_model.validate(data)

    def test_build_decision_schema_has_nullable_final_answer_and_tool_calls(self):
        schema = self._director().build_decision_schema()

        assert {"type": "null"} in schema["properties"]["final_answer"]["anyOf"]
        assert {"type": "null"} in schema["properties"]["tool_calls"]["anyOf"]
        assert schema["required"] == ["final_answer", "tool_calls"]

    def test_build_system_prompt_mentions_both_options(self):
        director = self._director()
        prompt = director.build_system_prompt_addition(director.build_decision_schema())

        assert "tool_calls" in prompt
        assert "final_answer" in prompt

    def test_process_decision_returns_tool_call_decision(self):
        director = self._director()
        decision = self._decision(
            director,
            {
                "final_answer": None,
                "tool_calls": [{"name": "search", "arguments": {"q": "x"}}],
            },
        )
        result = director.process_decision(decision)

        assert isinstance(result, ToolCallDecision)
        assert result.calls[0].name == "search"
        assert result.calls[0].arguments == {"q": "x"}

    def test_process_decision_returns_final_answer_decision(self):
        director = self._director(output_type=str)
        decision = self._decision(
            director, {"final_answer": "done", "tool_calls": None}
        )
        result = director.process_decision(decision)

        assert isinstance(result, FinalAnswerDecision)
        assert result.answer == "done"

    def test_process_decision_validates_final_answer_type(self):
        director = self._director(output_type=MyOutput)
        decision = self._decision(
            director, {"final_answer": {"name": "ok", "value": 7}, "tool_calls": None}
        )
        result = director.process_decision(decision)

        assert isinstance(result, FinalAnswerDecision)
        assert isinstance(result.answer, MyOutput)
        assert result.answer.value == 7

    def test_process_decision_raises_when_both_null(self):
        director = self._director()
        with pytest.raises(
            ValueError,
            match="tool_calls.*final_answer|final_answer.*tool_calls",
        ):
            self._decision(director, {"final_answer": None, "tool_calls": None})

    def test_decision_validation_rejects_both_tool_calls_and_final_answer(self):
        director = self._director()
        with pytest.raises(ValueError, match="must not contain both"):
            self._decision(
                director,
                {
                    "final_answer": "ignored",
                    "tool_calls": [{"name": "search", "arguments": {"q": "x"}}],
                },
            )


class TestOutputOnlyDirector:
    """Tests for _OutputOnlyDirector — no tools, final answer required."""

    def _director(self, output_type: Any = str):
        return _OutputOnlyDirector(PydanticModelInspector(), output_type, [])

    def test_build_decision_schema_has_only_final_answer(self):
        schema = self._director().build_decision_schema()

        assert "tool_calls" not in schema.get("properties", {})
        assert "final_answer" in schema["properties"]
        assert schema["required"] == ["final_answer"]

    def test_build_system_prompt_mentions_no_tools(self):
        director = self._director()
        prompt = director.build_system_prompt_addition(director.build_decision_schema())

        assert "No tools are available" in prompt

    def test_process_decision_returns_final_answer(self):
        director = self._director(output_type=str)
        decision = director.decision_model.validate({"final_answer": "hello"})
        result = director.process_decision(decision)

        assert isinstance(result, FinalAnswerDecision)
        assert result.answer == "hello"

    def test_process_decision_validates_structured_output(self):
        director = self._director(output_type=MyOutput)
        decision = director.decision_model.validate(
            {"final_answer": {"name": "test", "value": 99}}
        )
        result = director.process_decision(decision)

        assert isinstance(result, FinalAnswerDecision)
        assert isinstance(result.answer, MyOutput)
        assert result.answer.name == "test"

    def test_decision_model_rejects_null_final_answer(self):
        director = self._director()
        with pytest.raises(ValueError, match="Decision validation failed"):
            director.decision_model.validate({"final_answer": None})
