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
