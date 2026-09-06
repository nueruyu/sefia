from collections.abc import Callable
from typing import Any

from sefia import JsonSchemaToolEntry

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
