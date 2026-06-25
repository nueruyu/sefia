from typing import Annotated, Never

import pytest
from pydantic import Field

from sefia.exceptions import InvalidInferenceResponseError
from sefia.inference import ToolCallDecision
from sefia.llm._strategy import (
    LLMToolCall,
    _LLMDecision,
    _ToolEnabledDirector,
    _ToolOnlyDirector,
)
from sefia.pydantic import PydanticModelInspector


async def ask_user(question: Annotated[str, Field(min_length=1)]) -> str:
    """Ask the user a question and return the answer."""
    raise NotImplementedError


def _tool_schema() -> dict:
    return PydanticModelInspector().get_schema_for_function(ask_user)


def _tool_call_item(schema: dict) -> dict:
    items = schema["properties"]["tool_calls"]["items"]
    alternatives = items.get("anyOf")
    return alternatives[0] if alternatives else items


def test_tool_only_schema_embeds_tool_argument_schema() -> None:
    director = _ToolOnlyDirector(PydanticModelInspector(), Never, [_tool_schema()])

    schema = director.build_decision_schema()

    tool_calls = schema["properties"]["tool_calls"]
    assert tool_calls["minItems"] == 1
    item = _tool_call_item(schema)
    assert item["properties"]["name"]["enum"] == ["ask_user"]
    arguments = item["properties"]["arguments"]
    assert arguments["required"] == ["question"]
    assert arguments["additionalProperties"] is False
    assert arguments["properties"]["question"]["minLength"] == 1


def test_tool_enabled_schema_embeds_tool_argument_schema() -> None:
    director = _ToolEnabledDirector(PydanticModelInspector(), str, [_tool_schema()])

    schema = director.build_decision_schema()

    tool_calls_schema = schema["properties"]["tool_calls"]
    array_schema = next(
        candidate
        for candidate in tool_calls_schema["anyOf"]
        if candidate.get("type") == "array"
    )
    assert array_schema["minItems"] == 1
    item = array_schema["items"]
    assert item["properties"]["name"]["enum"] == ["ask_user"]
    assert item["properties"]["arguments"]["required"] == ["question"]


def test_tool_call_validation_rejects_unknown_tool() -> None:
    director = _ToolOnlyDirector(PydanticModelInspector(), Never, [_tool_schema()])

    with pytest.raises(InvalidInferenceResponseError, match="unknown tool"):
        director.process_decision(
            _LLMDecision(
                tool_calls=[LLMToolCall(name="unknown", arguments={"question": "Hi"})]
            )
        )


def test_tool_call_validation_rejects_missing_required_argument() -> None:
    director = _ToolOnlyDirector(PydanticModelInspector(), Never, [_tool_schema()])

    with pytest.raises(InvalidInferenceResponseError, match="question"):
        director.process_decision(
            _LLMDecision(tool_calls=[LLMToolCall(name="ask_user", arguments={})])
        )


def test_tool_call_validation_rejects_empty_min_length_argument() -> None:
    director = _ToolOnlyDirector(PydanticModelInspector(), Never, [_tool_schema()])

    with pytest.raises(InvalidInferenceResponseError, match="at least 1"):
        director.process_decision(
            _LLMDecision(
                tool_calls=[LLMToolCall(name="ask_user", arguments={"question": ""})]
            )
        )


def test_tool_call_validation_accepts_valid_arguments() -> None:
    director = _ToolOnlyDirector(PydanticModelInspector(), Never, [_tool_schema()])

    result = director.process_decision(
        _LLMDecision(
            tool_calls=[LLMToolCall(name="ask_user", arguments={"question": "Hello"})]
        )
    )

    assert isinstance(result, ToolCallDecision)
    assert result.calls[0].name == "ask_user"
    assert result.calls[0].arguments == {"question": "Hello"}
