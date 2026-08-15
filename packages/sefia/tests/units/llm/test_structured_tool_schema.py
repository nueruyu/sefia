import json
from typing import Annotated, Any, Literal, Never
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import Field, TypeAdapter, ValidationError

from sefia._interfaces import DecisionModelSpec
from sefia._tool_system import SignatureToolEntry, ToolEntry, ToolRegistry
from sefia.event_system import Event, EventPublisher
from sefia.exceptions import InvalidInferenceResponseError, UnknownToolDecisionError
from sefia.inference import FunctionInfo, InferenceDecision, ToolCallDecision
from sefia.llm import LLMInferenceStrategy, LLMResponse
from sefia.llm._execution_directors import ToolOnlyDirector
from sefia.pydantic import PydanticModelBackend
from sefia.pydantic._decision_model import _unknown_tool_name_from_error


async def ask_user(question: Annotated[str, Field(min_length=1)]) -> str:
    """Ask the user a question and return the answer."""
    raise NotImplementedError


class _MockPublisher(EventPublisher):
    def __init__(self) -> None:
        super().__init__(handlers=[])

    async def publish(self, event: Event) -> None:
        pass


def _tool() -> ToolEntry:
    backend = PydanticModelBackend()
    name = backend.tool_name(ask_user)
    return SignatureToolEntry(
        ask_user,
        name=name,
        schema_source=ask_user,
        inspector=backend,
    )


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follow a top-level ``$ref`` into ``$defs`` so assertions can inspect the
    embedded per-tool schemas regardless of how Pydantic hoists definitions."""
    if "$ref" in schema:
        key = schema["$ref"].split("/")[-1]
        return root["$defs"][key]
    return schema


def _tool_calls_array(schema: dict[str, Any]) -> dict[str, Any]:
    tool_calls = schema["properties"]["tool_calls"]
    if "anyOf" in tool_calls:
        return next(
            candidate
            for candidate in tool_calls["anyOf"]
            if candidate.get("type") == "array"
        )
    return tool_calls


def _tool_call_item(schema: dict[str, Any]) -> dict[str, Any]:
    return _resolve(_tool_calls_array(schema)["items"], schema)


def _name_constraint(name_schema: dict[str, Any]) -> Any:
    # A single Literal renders as `const`; multiple values render as `enum`.
    if "const" in name_schema:
        return name_schema["const"]
    return name_schema.get("enum")


def test_tool_only_schema_embeds_tool_argument_schema() -> None:
    director = ToolOnlyDirector(PydanticModelBackend(), Never, [_tool()])

    schema = director.build_decision_schema()

    assert _tool_calls_array(schema)["minItems"] == 1
    item = _tool_call_item(schema)
    assert _name_constraint(item["properties"]["name"]) in ("ask_user", ["ask_user"])
    arguments = _resolve(item["properties"]["arguments"], schema)
    assert arguments["required"] == ["question"]
    assert arguments["additionalProperties"] is False
    assert arguments["properties"]["question"]["minLength"] == 1


def test_unknown_tool_name_ignores_root_literal_errors() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(Literal["expected"]).validate_python("actual")

    assert _unknown_tool_name_from_error(exc_info.value) is None


def test_decision_model_spec_rejects_tool_modes_without_tools() -> None:
    with pytest.raises(ValueError, match="require at least one tool"):
        DecisionModelSpec.tool_only(
            name="LLMDecision",
            output_type=Never,
            tools=[],
        )

    with pytest.raises(ValueError, match="require at least one tool"):
        DecisionModelSpec.tool_enabled(
            name="LLMDecision",
            output_type=str,
            tools=[],
        )


class TestToolCallValidation:
    """The decision model validates tool arguments end-to-end via the backend."""

    def _strategy(self, content: str) -> LLMInferenceStrategy:
        client = AsyncMock()
        client.complete.return_value = LLMResponse(content=content)
        formatter = Mock()
        formatter.format_arguments.return_value = "<arguments/>"
        return LLMInferenceStrategy(
            llm_client=client,
            decision_builder=PydanticModelBackend(),
            prompt_formatter=formatter,
        )

    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.add(ask_user, name="ask_user")
        return registry

    async def _decide(self, payload: dict[str, Any]) -> InferenceDecision:
        strategy = self._strategy(json.dumps(payload))
        return await strategy.decide_next_step(
            FunctionInfo(
                qualname="test",
                instructions="chat",
                bound_arguments={},
                type_hints={},
                return_type=Never,
                args=(),
                kwargs={},
            ),
            [],
            self._registry(),
            _MockPublisher(),
        )

    async def test_rejects_unknown_tool_with_specific_cause(self) -> None:
        with pytest.raises(InvalidInferenceResponseError) as exc_info:
            await self._decide(
                {
                    "decision": "tool_calls",
                    "tool_calls": [{"name": "unknown", "arguments": {}}],
                }
            )

        assert isinstance(exc_info.value.__cause__, UnknownToolDecisionError)
        assert exc_info.value.__cause__.tool_name == "unknown"

    async def test_rejects_missing_required_argument(self) -> None:
        with pytest.raises(InvalidInferenceResponseError, match="question"):
            await self._decide(
                {
                    "decision": "tool_calls",
                    "tool_calls": [{"name": "ask_user", "arguments": {}}],
                }
            )

    async def test_rejects_empty_min_length_argument(self) -> None:
        with pytest.raises(InvalidInferenceResponseError, match="non-empty"):
            await self._decide(
                {
                    "decision": "tool_calls",
                    "tool_calls": [{"name": "ask_user", "arguments": {"question": ""}}],
                }
            )

    async def test_rejects_unknown_argument(self) -> None:
        with pytest.raises(InvalidInferenceResponseError, match="LLM output failed"):
            await self._decide(
                {
                    "decision": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "ask_user",
                            "arguments": {"question": "Hi", "extra": 1},
                        }
                    ],
                }
            )

    async def test_accepts_valid_arguments(self) -> None:
        decision = await self._decide(
            {
                "decision": "tool_calls",
                "tool_calls": [
                    {"name": "ask_user", "arguments": {"question": "Hello"}}
                ],
            }
        )

        assert isinstance(decision, ToolCallDecision)
        assert decision.calls[0].name == "ask_user"
        assert decision.calls[0].arguments == {"question": "Hello"}
