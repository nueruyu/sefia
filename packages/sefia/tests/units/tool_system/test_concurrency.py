from typing import Any, Protocol

from sefia import ToolEntry, ToolRegistry, Tools, concurrent
from sefia.inference import Capability
from sefia.tool_collectors import DefaultToolCollector


def _collect_self(instance: object) -> ToolRegistry:
    return DefaultToolCollector().collect([Capability(value=instance, declared=None)])


def _tool_named(registry: ToolRegistry, suffix: str) -> ToolEntry:
    # Locally-defined toolkit classes get their qualname sanitized into the
    # tool name; match on the method-name suffix instead of the full name.
    (tool,) = [t for t in registry.get_all() if t.name.endswith(suffix)]
    return tool


def test_concurrent_marker_is_collected():
    class Toolkit:
        @concurrent
        async def search(self, query: str) -> str:
            """Search."""
            return query

        async def write(self, text: str) -> None:
            """Write."""

    class Agent:
        _kit: Tools[Toolkit]

        def __init__(self):
            self._kit = Toolkit()

    registry = _collect_self(Agent())

    assert _tool_named(registry, "_search").concurrent is True
    assert _tool_named(registry, "_write").concurrent is False


def test_concurrent_marker_on_static_and_class_tools():
    class Toolkit:
        @concurrent
        @staticmethod
        async def static_tool() -> str:
            """Static."""
            return "s"

        @concurrent
        @classmethod
        async def class_tool(cls) -> str:
            """Class."""
            return "c"

    class Agent:
        _kit: Tools[Toolkit]

        def __init__(self):
            self._kit = Toolkit()

    registry = _collect_self(Agent())

    assert _tool_named(registry, "_static_tool").concurrent is True
    assert _tool_named(registry, "_class_tool").concurrent is True


def test_concurrent_marker_is_read_from_the_implementation_under_protocol_narrowing():
    # The marker describes the concrete implementation's runtime behavior, so
    # it is read from the implementation even when the schema comes from a
    # Protocol whose own method is unmarked.
    class ReadOnlyWeb(Protocol):
        async def search(self, q: str) -> list[str]:
            """Search the web."""
            ...

    class BroadWebClient:
        @concurrent
        async def search(self, q: str) -> list[str]:
            return [q]

    class Agent:
        _web: Tools[ReadOnlyWeb]

        def __init__(self, web: ReadOnlyWeb):
            self._web = web

    registry = _collect_self(Agent(BroadWebClient()))

    assert _tool_named(registry, "_search").concurrent is True


def test_registry_tools_default_to_serial():
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
