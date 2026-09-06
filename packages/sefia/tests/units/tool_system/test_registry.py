from typing import Any

import pytest
from sefia import ToolRegistry
from sefia.exceptions import ToolConflictError


def tool_function() -> str:
    return "tool"


class ExampleTool:
    def method(self) -> str:
        return "method"


def test_get_by_function_returns_all_registered_tools_for_function():
    registry = ToolRegistry()
    registry.add(tool_function, name="first")
    registry.add(tool_function, name="second")
    first = registry.get("first")
    second = registry.get("second")

    assert first is not None
    assert second is not None
    assert registry.get_by_function(tool_function) == [first, second]


def test_get_by_function_matches_bound_method_against_unbound_function():
    registry = ToolRegistry()
    registry.add(ExampleTool().method, name="method")
    method = registry.get("method")

    assert method is not None
    assert registry.get_by_function(ExampleTool.method) == [method]


def test_tools_default_to_serial() -> None:
    async def handler(**kwargs: Any) -> str:
        return "ok"

    registry = ToolRegistry()
    registry.add(handler, name="plain")
    registry.add_json_tool(
        handler, name="json_plain", description="", parameters={"type": "object"}
    )
    registry.add(handler, name="marked", concurrent=True)

    plain = registry.get("plain")
    json_plain = registry.get("json_plain")
    marked = registry.get("marked")
    assert plain is not None and plain.concurrent is False
    assert json_plain is not None and json_plain.concurrent is False
    assert marked is not None and marked.concurrent is True


_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def _noop(**kwargs: Any) -> None:
    pass


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
