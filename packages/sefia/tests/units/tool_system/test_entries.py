import jsonschema
import pytest
from collections.abc import Callable
from typing import Any, Never

from sefia import JsonSchemaToolEntry, ToolRegistry
from sefia.exceptions import ToolConflictError
from sefia.inference import ToolCallsDecision
from sefia.llm._tool_call_ids import ToolCallIdRegistry
from sefia.llm.step_decision import StepDecisionModel, StepDecisionSpec
from sefia.pydantic import PydanticModelBackend

_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def _search_tool(handler: Callable[..., Any]) -> JsonSchemaToolEntry:
    return JsonSchemaToolEntry(
        handler,
        name="search",
        parameters=_SEARCH_SCHEMA,
        description="Search the corpus.",
    )


def _noop(**kwargs: Any) -> None:
    pass


def _argument_names(**kwargs: Any) -> list[str]:
    return list(kwargs)


def test_definition_is_the_raw_json_schema_verbatim():
    tool = _search_tool(_noop)

    definition = tool.definition()

    assert definition.name == "search"
    assert definition.description == "Search the corpus."
    # The JSON Schema reaches the LLM verbatim, with no signature introspection.
    assert definition.parameters is _SEARCH_SCHEMA
    assert definition.to_dict() == {
        "name": "search",
        "description": "Search the corpus.",
        "parameters": _SEARCH_SCHEMA,
    }


async def test_invoke_dispatches_decoded_arguments_to_the_handler():
    received: dict[str, Any] = {}

    async def handler(**kwargs: Any) -> str:
        received.update(kwargs)
        return "hits"

    tool = _search_tool(handler)

    result = await tool.invoke({"query": "sefia", "limit": 3})

    assert result == "hits"
    assert received == {"query": "sefia", "limit": 3}


async def test_invoke_supports_a_synchronous_handler():
    tool = _search_tool(_argument_names)

    assert await tool.invoke({"query": "x"}) == ["query"]


def test_registration_shares_the_namespace_with_introspected_tools():
    def existing_search(query: str) -> str:
        """A signature-based tool."""
        raise NotImplementedError

    registry = ToolRegistry()
    registry.add(existing_search, name="search")

    with pytest.raises(ToolConflictError):
        registry.add_json_tool(
            _noop,
            name="search",
            description="dup",
            parameters=_SEARCH_SCHEMA,
        )


def test_a_malformed_schema_is_rejected_up_front():
    tool = JsonSchemaToolEntry(
        _noop,
        name="invalid",
        parameters={"type": "not-a-type"},
    )
    with pytest.raises(jsonschema.SchemaError):
        StepDecisionModel.from_spec(
            StepDecisionSpec.for_inference(
                name="StepDecision", output_type=Never, tools=[tool]
            ),
            PydanticModelBackend(),
        )


def test_a_schema_is_validated_under_its_declared_dialect():
    # Array-form ``items`` (tuple validation) is draft-07; under the default
    # 2020-12 dialect it would be rejected as malformed.
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "pair": {"items": [{"type": "string"}, {"type": "integer"}]},
        },
        "required": ["pair"],
    }

    tool = JsonSchemaToolEntry(_noop, name="pair", parameters=schema)
    decision_model = StepDecisionModel.from_spec(
        StepDecisionSpec.for_inference(
            name="StepDecision", output_type=Never, tools=[tool]
        ),
        PydanticModelBackend(),
    )
    tool_call_ids = ToolCallIdRegistry()

    valid = decision_model.validate(
        {
            "decision": "tool_calls",
            "tool_calls": [{"name": "pair", "arguments": {"pair": ["a", 1]}}],
        },
        tool_call_ids,
    )
    assert isinstance(valid, ToolCallsDecision)
    assert valid.calls[0].arguments == {"pair": ["a", 1]}

    with pytest.raises(ValueError, match="Step decision validation failed"):
        decision_model.validate(
            {
                "decision": "tool_calls",
                "tool_calls": [{"name": "pair", "arguments": {"pair": [1, "a"]}}],
            },
            tool_call_ids,
        )
