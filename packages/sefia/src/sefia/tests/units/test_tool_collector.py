import functools

import pytest

from sefia import infer, tool, toolify
from sefia.models import ToolConflictError
from sefia.tool_collectors.collector import DefaultToolCollector
from sefia.toolify import Toolset

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


class GreetingToolkit:
    @tool
    async def greet(self, name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}"


class MyAgent:
    def __init__(self, toolkit: WebToolkit):
        self._toolkit = toolkit  # private member → its @tool methods are scanned
        self.greeter = GreetingToolkit()  # public member → also scanned


def test_collect_tools_from_public_and_private_members():
    agent = MyAgent(WebToolkit())
    collector = DefaultToolCollector()
    registry = collector.collect(agent)

    tool_names = set(registry.tools.keys())
    # Marked methods held in a private attribute.
    assert "WebToolkit_search" in tool_names
    assert "WebToolkit_fetch_content" in tool_names
    # Marked method held in a public attribute.
    assert "GreetingToolkit_greet" in tool_names
    assert len(tool_names) == 3


class ConflictingAgent:
    def __init__(self):
        self._kit1 = WebToolkit()
        self._kit2 = WebToolkit()  # same tool names → conflict


def test_collect_tools_with_conflict_raises_error():
    agent = ConflictingAgent()
    collector = DefaultToolCollector()
    with pytest.raises(ToolConflictError):
        collector.collect(agent)


class SelfMethodAgent:
    """Exposes its own @tool methods, including private ones, but not @infer."""

    def __init__(self):
        self._note = "held primitive, not a tool provider"

    @tool
    async def marked_public(self, value: int) -> int:
        """A marked public helper exposed as a tool."""
        return value

    @tool
    async def _marked_private(self, value: int) -> int:
        """A marked private helper, still exposed for the instance itself."""
        return value

    async def unmarked(self, value: int) -> int:
        """Not marked, so not a tool."""
        return value

    @infer()
    async def run(self, task: str) -> str:
        """An inference entry point, not a tool."""
        ...


def test_collect_exposes_marked_methods_including_private_but_not_infer():
    collector = DefaultToolCollector()
    registry = collector.collect(SelfMethodAgent())

    tool_names = set(registry.tools.keys())
    assert "SelfMethodAgent_marked_public" in tool_names
    # The leading "." of "._marked_private" sanitizes to a second underscore.
    assert "SelfMethodAgent__marked_private" in tool_names
    # Unmarked methods and @infer entry points are not tools.
    assert not any(name.endswith("_unmarked") for name in tool_names)
    assert not any(name.endswith("_run") for name in tool_names)
    # A held primitive (str) must not leak its public methods as tools.
    assert not any("upper" in name for name in tool_names)
    assert len(tool_names) == 2


class ExternalLikeClient:
    """Simulates a third-party class we cannot decorate with @tool."""

    class Error(Exception):
        """A nested class attribute — callable, but not a tool."""

    async def fetch(self, url: str) -> str:
        """Fetch a URL."""
        return url

    def _internal(self) -> None:
        ...


async def standalone_search(query: str) -> str:
    """A standalone search function."""
    return query


def test_toolify_bundles_public_methods_and_functions():
    box = toolify(ExternalLikeClient(), standalone_search)
    assert isinstance(box, Toolset)
    # One public method of the object plus the standalone function.
    assert len(box.tools) == 2


class ToolifyAgent:
    def __init__(self):
        self._tools = toolify(ExternalLikeClient(), standalone_search)


def test_collect_registers_toolify_members():
    collector = DefaultToolCollector()
    registry = collector.collect(ToolifyAgent())

    tool_names = set(registry.tools.keys())
    assert "ExternalLikeClient_fetch" in tool_names
    assert "standalone_search" in tool_names
    # The external object's private method is not exposed.
    assert not any("internal" in name for name in tool_names)
    assert len(tool_names) == 2


def test_toolify_keeps_partial_and_skips_builtins():
    bound = functools.partial(standalone_search, "fixed")
    # A str/list are builtin instances and must not leak their methods.
    box = toolify(bound, "a string", [1, 2])
    # Only the partial is registered, with its bound argument intact.
    assert box.tools == [bound]


class StaticToolHost:
    @tool
    @staticmethod
    async def static_tool(value: int) -> int:
        """A static tool."""
        return value

    @tool
    @classmethod
    async def class_tool(cls, value: int) -> int:
        """A class tool."""
        return value


def test_tool_marks_static_and_class_methods():
    collector = DefaultToolCollector()
    registry = collector.collect(StaticToolHost())

    tool_names = set(registry.tools.keys())
    assert any(name.endswith("static_tool") for name in tool_names)
    assert any(name.endswith("class_tool") for name in tool_names)
