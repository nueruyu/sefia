import pytest

from sefia.models import ToolConflictError
from sefia.tool_collectors.collector import DefaultToolCollector

from ..conftest import WebToolkit


def example_func(a: int, b: str = "default") -> bool:
    """An example function."""
    return True


def test_create_tool_schema_from_function():
    collector = DefaultToolCollector()
    # Test the private schema building method directly
    schema = collector._build_schema(example_func)

    assert schema["type"] == "function"
    function_spec = schema["function"]

    assert function_spec["name"] == "example_func"
    assert function_spec["description"] == "An example function."

    params = function_spec["parameters"]
    assert params["type"] == "object"
    assert "a" in params["properties"]
    assert "b" in params["properties"]
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "string"
    assert params["properties"]["b"]["default"] == "default"
    assert "a" in params["required"]
    assert "b" not in params.get("required", [])


def test_schema_builder_sanitizes_complex_names():
    collector = DefaultToolCollector()

    class Outer:
        class Inner:
            def my_method(self):
                pass

    schema = collector._build_schema(Outer.Inner.my_method)
    name = schema["function"]["name"]
    # __qualname__ includes enclosing scope; verify it ends with the expected suffix
    assert name.endswith("Outer_Inner_my_method")
    # verify no dots or other unsafe characters remain
    assert "." not in name
    assert "<" not in name


def test_schema_builder_caches_results():
    collector = DefaultToolCollector()
    schema1 = collector._build_schema(example_func)
    schema2 = collector._build_schema(example_func)
    assert schema1 is schema2


class MyAgent:
    def __init__(self, toolkit: WebToolkit):
        self._toolkit = toolkit  # private → scanned as toolkit
        self.non_exposed_toolkit = toolkit  # public → not scanned as toolkit


def test_collect_tools_from_instance():
    agent = MyAgent(WebToolkit())
    collector = DefaultToolCollector()
    registry = collector.collect(agent)

    tool_names = set(registry.tools.keys())
    assert "WebToolkit_search" in tool_names
    assert "WebToolkit_fetch_content" in tool_names
    assert len(tool_names) == 2


class ConflictingAgent:
    def __init__(self):
        self._kit1 = WebToolkit()
        self._kit2 = WebToolkit()  # same tool names → conflict


def test_collect_tools_with_conflict_raises_error():
    agent = ConflictingAgent()
    collector = DefaultToolCollector()
    with pytest.raises(ToolConflictError):
        collector.collect(agent)
