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
    ToolCallsDecision,
)
from sefia.llm import LLMClient, LLMInferenceStrategy, LLMResponse
from sefia.llm._step_decision_prompt import build_step_decision_prompt
from sefia.llm._tool_call_ids import ToolCallIdRegistry
from sefia.llm.json_schema import SchemaNode
from sefia.llm.step_decision import (
    DefaultStepDecisionSchemaFactory,
    StepDecisionSchema,
    StepDecisionSpec,
)
from sefia.llm.structured_output import StructuredValue
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


@dataclass(frozen=True)
class _StepDecisionFixture:
    spec: StepDecisionSpec
    decision_schema: StepDecisionSchema

    @property
    def schema(self):
        return self.decision_schema.structured_output

    def prompt(self) -> str:
        return build_step_decision_prompt(self.spec)

    def validate(
        self,
        data: StructuredValue,
        tool_call_ids: ToolCallIdRegistry | None = None,
    ):
        return self.decision_schema.validate(data, tool_call_ids)


def _step(output_type: Any, tools: list[ToolEntry]) -> _StepDecisionFixture:
    spec = StepDecisionSpec.for_inference(
        name="StepDecision", output_type=output_type, tools=tools
    )
    return _StepDecisionFixture(
        spec, DefaultStepDecisionSchemaFactory(_BACKEND).create(spec)
    )


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
        step_decision_schema_factory=DefaultStepDecisionSchemaFactory(
            PydanticModelBackend()
        ),
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


class TestToolsRequiredDecision:
    """Tests for __StepDecisionFixture — the Never return type mode."""

    def test_step_returns_tool_only_for_never(self):
        step = _step(Never, [_tool(chat_tool)])
        assert isinstance(step, _StepDecisionFixture)

    def test_step_raises_for_never_without_tools(self):
        with pytest.raises(ValueError, match="must have tools available"):
            _step(Never, [])

    def test_structured_output_schema_has_no_result_field(self):
        step = _step(Never, [_tool(chat_tool)])
        schema = step.schema.document.to_dict()
        branch = _decision_branch(schema, "tool_calls")

        assert schema["required"] == ["decision", "tool_calls"]
        assert schema["additionalProperties"] is False
        assert "result" not in branch["properties"]
        assert branch["properties"]["decision"]["const"] == "tool_calls"
        assert "tool_calls" in branch["properties"]
        assert branch["required"] == ["decision", "tool_calls"]

    def test_build_system_prompt_instructs_tool_only(self):
        step = _step(Never, [_tool(chat_tool)])
        prompt = step.prompt()

        assert "result" not in prompt or "There is no `result`" in prompt
        assert "tool_calls" in prompt
        assert '"$defs"' not in prompt

    def test_process_decision_accepts_tool_calls(self):
        step = _step(Never, [_tool(chat_tool)])

        data: StructuredValue = {
            "decision": "tool_calls",
            "tool_calls": [{"name": "chat_tool", "arguments": {}}],
        }
        result = step.validate(data, ToolCallIdRegistry())

        assert isinstance(result, ToolCallsDecision)
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

        assert isinstance(result, ToolCallsDecision)

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


class TestToolsOrResultDecision:
    """Tests for __StepDecisionFixture — tools and result are both allowed."""

    def _step(self, output_type: Any = str) -> _StepDecisionFixture:
        return _step(output_type, [_tool(search)])

    def test_structured_output_schema_has_decision_branches(self):
        schema = self._step().schema.document.to_dict()
        payload = SchemaNode(schema)

        discriminator = payload.object_map("discriminator")
        assert discriminator is not None
        assert discriminator["propertyName"] == "decision"
        assert len(payload.one_of()) == 2
        assert _decision_branch(schema, "tool_calls")["required"] == [
            "decision",
            "tool_calls",
        ]
        assert _decision_branch(schema, "result")["required"] == [
            "decision",
            "result",
        ]

    def test_build_system_prompt_mentions_both_options(self):
        step = self._step()
        prompt = step.prompt()

        assert "tool_calls" in prompt
        assert "result" in prompt

    def test_process_decision_returns_tool_call_decision(self):
        step = self._step()
        result = step.validate(
            {
                "decision": "tool_calls",
                "tool_calls": [{"name": "search", "arguments": {"q": "x"}}],
            },
            ToolCallIdRegistry(),
        )

        assert isinstance(result, ToolCallsDecision)
        assert result.calls[0].name == "search"
        assert result.calls[0].arguments == {"q": "x"}

    def test_process_decision_returns_result_decision(self):
        step = self._step(output_type=str)
        result = step.validate({"decision": "result", "result": "done"})

        assert isinstance(result, ResultDecision)
        assert result.result == "done"

    def test_process_decision_accepts_logical_decision(self):
        step = self._step(output_type=str)

        result = step.validate({"decision": "result", "result": "done"})

        assert isinstance(result, ResultDecision)
        assert result.result == "done"

    def test_process_decision_validates_result_type(self):
        step = self._step(output_type=MyOutput)
        result = step.validate(
            {
                "decision": "result",
                "result": {"name": "ok", "value": 7},
            }
        )

        assert isinstance(result, ResultDecision)
        assert isinstance(result.result, MyOutput)
        assert result.result.value == 7

    def test_process_decision_raises_when_decision_branch_is_incomplete(self):
        step = self._step()
        with pytest.raises(
            ValueError,
            match="Step decision validation failed",
        ):
            step.validate({"decision": "tool_calls"})

    def test_decision_validation_rejects_fields_from_other_branch(self):
        step = self._step()
        with pytest.raises(ValueError, match="Step decision validation failed"):
            step.validate(
                {
                    "decision": "tool_calls",
                    "result": "extra",
                    "tool_calls": [{"name": "search", "arguments": {"q": "x"}}],
                },
            )


class TestResultOnlyDecision:
    """Tests for __StepDecisionFixture — no tools, result required."""

    def _step(self, output_type: Any = str) -> _StepDecisionFixture:
        return _step(output_type, [])

    def test_structured_output_schema_has_only_result(self):
        schema = self._step().schema.document.to_dict()
        branch = _decision_branch(schema, "result")

        assert schema["required"] == ["decision", "result"]
        assert "tool_calls" not in branch["properties"]
        assert "result" in branch["properties"]
        assert branch["properties"]["decision"]["const"] == "result"
        assert branch["required"] == ["decision", "result"]

    def test_build_system_prompt_mentions_no_tools(self):
        step = self._step()
        prompt = step.prompt()

        assert "No tools are available" in prompt

    def test_process_decision_returns_result(self):
        step = self._step(output_type=str)
        result = step.validate({"decision": "result", "result": "hello"})

        assert isinstance(result, ResultDecision)
        assert result.result == "hello"

    def test_process_decision_validates_structured_output(self):
        step = self._step(output_type=MyOutput)
        result = step.validate(
            {
                "decision": "result",
                "result": {"name": "test", "value": 99},
            }
        )

        assert isinstance(result, ResultDecision)
        assert isinstance(result.result, MyOutput)
        assert result.result.name == "test"

    def test_step_decision_rejects_null_result(self):
        step = self._step()
        with pytest.raises(ValueError, match="Step decision validation failed"):
            step.validate({"decision": "result", "result": None})
