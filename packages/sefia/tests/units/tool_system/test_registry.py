from typing import Any

from sefia import ToolRegistry


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
