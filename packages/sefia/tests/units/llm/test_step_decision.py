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
from sefia.llm import LLMClient, LLMInferenceStrategy, LLMCompletion
from sefia.llm._tool_call_ids import ToolCallIdRegistry
from sefia.llm.step_decision import DecisionSpec, StepDecisionMode
from sefia.llm.transports import StructuredDecisionTransport
from sefia.llm.json_schema import JsonValue
from sefia.llm.structured_data import StructuredData
from sefia.pydantic import PydanticModelBackend


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
    decision: DecisionSpec

    def validate(
        self,
        data: StructuredData | JsonValue,
        tool_call_ids: ToolCallIdRegistry | None = None,
    ):
        value = (
            data if isinstance(data, StructuredData) else StructuredData.from_json(data)
        )
        return self.decision.validate(value, tool_call_ids)


def _step(output_type: Any, tools: list[ToolEntry]) -> _StepDecisionFixture:
    return _StepDecisionFixture(
        DecisionSpec.for_inference(
            output_type=output_type,
            tools=tools,
            result_format_factory=_BACKEND,
        )
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
    """The strategy under test, with a stub prompt renderer."""
    renderer = Mock()
    renderer.render.return_value = "prompt"
    client = llm_client if llm_client is not None else AsyncMock()
    return LLMInferenceStrategy(
        llm_client=client,
        result_format_factory=PydanticModelBackend(),
        prompt_renderer=renderer,
        decision_transport=StructuredDecisionTransport(),
        stream=stream,
        max_repair_attempts=max_repair_attempts,
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


class TestToolsRequiredDecision:
    """Tests for __StepDecisionFixture — the Never return type mode."""

    def test_step_returns_tool_only_for_never(self):
        step = _step(Never, [_tool(chat_tool)])
        assert isinstance(step, _StepDecisionFixture)

    def test_step_raises_for_never_without_tools(self):
        with pytest.raises(ValueError, match="must have tools available"):
            _step(Never, [])

    def test_model_exposes_tool_only_structure(self):
        step = _step(Never, [_tool(chat_tool)])
        assert step.decision.mode is StepDecisionMode.TOOLS_REQUIRED
        assert step.decision.result is None
        assert [tool.name for tool in step.decision.tools] == ["chat_tool"]

    def test_process_decision_accepts_tool_calls(self):
        step = _step(Never, [_tool(chat_tool)])

        data = StructuredData.from_json(
            {
                "decision": "tool_calls",
                "tool_calls": [{"name": "chat_tool", "arguments": {}}],
            }
        )
        result = step.validate(data, ToolCallIdRegistry())

        assert isinstance(result, ToolCallsDecision)
        assert result.calls[0].name == "chat_tool"

    async def test_decide_next_step_returns_tool_call_decision_for_never(self):
        mock_client = AsyncMock()
        content = (
            '{"decision": "tool_calls", '
            '"tool_calls": [{"name": "chat_tool", "arguments": {}}]}'
        )
        mock_client.complete.return_value = LLMCompletion(
            content=content,
            structured_output=StructuredData.parse_json(content),
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
        content = '{"decision": "result", "result": "bye"}'
        mock_client.complete.return_value = LLMCompletion(
            content=content,
            structured_output=StructuredData.parse_json(content),
        )
        strategy = _make_strategy(mock_client)

        with pytest.raises(
            InvalidInferenceResponseError, match="LLM decision failed validation"
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

    def test_model_exposes_tools_and_result(self):
        model = self._step().decision
        assert model.mode is StepDecisionMode.TOOLS_OR_RESULT
        assert model.result is not None
        assert [tool.name for tool in model.tools] == ["search"]

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

    def test_model_exposes_result_only_structure(self):
        model = self._step().decision
        assert model.mode is StepDecisionMode.RESULT_ONLY
        assert model.result is not None
        assert model.tools == ()

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
