from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Never
from unittest.mock import AsyncMock, Mock

import pytest

from sefia._tool_system import SignatureToolEntry, ToolEntry, ToolRegistry
from sefia.event_system import Event, EventPublisher
from sefia.exceptions import InvalidInferenceResponseError
from sefia.inference import (
    FunctionInfo,
    ResultDecision,
    ToolCallDecision,
)
from sefia.llm import LLMClient, LLMInferenceStrategy, LLMResponse
from sefia.llm._execution_directors import (
    OutputOnlyDirector,
    ToolEnabledDirector,
    ToolOnlyDirector,
)
from sefia.llm._tool_call_ids import ToolCallIdRegistry
from sefia.llm.schema import SchemaNode
from sefia.pydantic import PydanticModelBackend
from sefia.pydantic._json_utils import pydantic_json_default


class MockEventPublisher(EventPublisher):
    def __init__(self) -> None:
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


def _make_strategy(
    llm_client: LLMClient | None = None,
    *,
    stream: bool = False,
    max_repair_attempts: int = 2,
) -> LLMInferenceStrategy:
    """The strategy under test, with a stub prompt formatter."""
    formatter = Mock()
    formatter.format_arguments.return_value = "<arguments/>"
    client = llm_client if llm_client is not None else AsyncMock()
    return LLMInferenceStrategy(
        llm_client=client,
        decision_builder=PydanticModelBackend(),
        prompt_formatter=formatter,
        json_default=pydantic_json_default,
        stream=stream,
        max_repair_attempts=max_repair_attempts,
    )


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in schema:
        key = schema["$ref"].split("/")[-1]
        return root["$defs"][key]
    return schema


def _decision_branch(schema: dict[str, Any], decision: str) -> dict[str, Any]:
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


class TestToolOnlyDirector:
    """Tests for _ToolOnlyDirector — the Never return type mode."""

    def test_create_director_returns_tool_only_for_never(self):
        strategy = _make_strategy()
        director = strategy._create_director(Never, [_tool(chat_tool)])
        assert isinstance(director, ToolOnlyDirector)

    def test_create_director_raises_for_never_without_tools(self):
        strategy = _make_strategy()
        with pytest.raises(ValueError, match="must have tools available"):
            strategy._create_director(Never, [])

    def test_build_decision_schema_has_no_result_field(self):
        strategy = _make_strategy()
        director = strategy._create_director(Never, [_tool(chat_tool)])
        schema = director.build_decision_schema().document.to_dict()
        branch = _decision_branch(schema, "tool_calls")

        assert schema["required"] == ["decision", "tool_calls"]
        assert schema["additionalProperties"] is False
        assert "result" not in branch["properties"]
        assert branch["properties"]["decision"]["const"] == "tool_calls"
        assert "tool_calls" in branch["properties"]
        assert branch["required"] == ["decision", "tool_calls"]

    def test_build_system_prompt_instructs_tool_only(self):
        strategy = _make_strategy()
        director = strategy._create_director(Never, [_tool(chat_tool)])
        prompt = director.build_system_prompt_addition()

        assert "result" not in prompt or "There is no `result`" in prompt
        assert "tool_calls" in prompt
        assert '"$defs"' not in prompt

    def test_process_decision_accepts_tool_calls(self):
        strategy = _make_strategy()
        director = strategy._create_director(Never, [_tool(chat_tool)])

        data: object = {
            "decision": "tool_calls",
            "tool_calls": [{"name": "chat_tool", "arguments": dict[str, object]()}],
        }
        result = director.process_response_data(data, ToolCallIdRegistry())

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
        strategy = _make_strategy(mock_client)

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
        strategy = _make_strategy(mock_client)

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

    def _director(self, output_type: Any = str) -> ToolEnabledDirector:
        return ToolEnabledDirector(PydanticModelBackend(), output_type, [_tool(search)])

    def test_build_decision_schema_has_decision_branches(self):
        schema = self._director().build_decision_schema().document.to_dict()
        payload = SchemaNode(schema)

        discriminator = payload.object_map("discriminator")
        assert discriminator is not None
        assert discriminator["propertyName"] == "decision"
        assert len(payload.alternatives("oneOf")) == 2
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
        prompt = director.build_system_prompt_addition()

        assert "tool_calls" in prompt
        assert "result" in prompt

    def test_process_decision_returns_tool_call_decision(self):
        director = self._director()
        result = director.process_response_data(
            {
                "decision": "tool_calls",
                "tool_calls": [{"name": "search", "arguments": {"q": "x"}}],
            },
            ToolCallIdRegistry(),
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

    def test_process_decision_accepts_logical_decision(self):
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

    def _director(self, output_type: Any = str) -> OutputOnlyDirector:
        return OutputOnlyDirector(PydanticModelBackend(), output_type, [])

    def test_build_decision_schema_has_only_result(self):
        schema = self._director().build_decision_schema().document.to_dict()
        branch = _decision_branch(schema, "result")

        assert schema["required"] == ["decision", "result"]
        assert "tool_calls" not in branch["properties"]
        assert "result" in branch["properties"]
        assert branch["properties"]["decision"]["const"] == "result"
        assert branch["required"] == ["decision", "result"]

    def test_build_system_prompt_mentions_no_tools(self):
        director = self._director()
        prompt = director.build_system_prompt_addition()

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
