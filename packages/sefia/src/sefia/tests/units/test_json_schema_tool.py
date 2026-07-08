import jsonschema
import pytest

from sefia import JsonSchemaTool, ToolRegistry
from sefia.exceptions import ToolConflictError
from sefia.pydantic._function_models import json_schema_argument_type

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def _search_tool(handler) -> JsonSchemaTool:
    return JsonSchemaTool(
        handler,
        name="search",
        parameters=_SEARCH_SCHEMA,
        description="Search the corpus.",
    )


def test_definition_is_the_raw_json_schema_verbatim():
    tool = _search_tool(lambda **kwargs: None)

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
    received: dict = {}

    async def handler(**kwargs):
        received.update(kwargs)
        return "hits"

    tool = _search_tool(handler)

    result = await tool.invoke({"query": "sefia", "limit": 3})

    assert result == "hits"
    assert received == {"query": "sefia", "limit": 3}


async def test_invoke_supports_a_synchronous_handler():
    tool = _search_tool(lambda **kwargs: list(kwargs))

    assert await tool.invoke({"query": "x"}) == ["query"]


def test_registration_shares_the_namespace_with_introspected_tools():
    def existing_search(query: str) -> str:
        """A signature-based tool."""
        raise NotImplementedError

    registry = ToolRegistry()
    registry.add(existing_search, name="search")

    with pytest.raises(ToolConflictError):
        registry.add_json_tool(
            lambda **kwargs: None,
            name="search",
            description="dup",
            parameters=_SEARCH_SCHEMA,
        )


def test_a_malformed_schema_is_rejected_up_front():
    with pytest.raises(jsonschema.SchemaError):
        json_schema_argument_type({"type": "not-a-type"})
