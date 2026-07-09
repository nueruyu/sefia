from typing import Sequence

from .._tool_system import Tool, ToolCollector, ToolRegistry


class StaticToolCollector(ToolCollector):
    """A collector that yields a fixed set of pre-built tools, ignoring the
    instance. The seam for injecting tools that have no Python instance to
    introspect (JSON-schema / client-side tools)."""

    def __init__(self, tools: Sequence[Tool]):
        self._tools = list(tools)

    def collect(self, instance: object) -> ToolRegistry:
        registry = ToolRegistry()
        for tool in self._tools:
            registry.register(tool)
        return registry
