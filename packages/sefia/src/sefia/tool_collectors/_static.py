from typing import Sequence

from typing_extensions import final, override

from .._tool_system import ToolEntry, ToolCollector, ToolRegistry
from ..inference import Capability


@final
class StaticToolCollector(ToolCollector):
    """A collector that yields a fixed set of pre-built tools, ignoring the
    call's capability parameters. The seam for injecting tools that have no
    Python instance to introspect (JSON-schema / client-side tools)."""

    def __init__(self, tools: Sequence[ToolEntry]):
        self._tools = list(tools)

    @override
    def collect(self, capabilities: list[Capability]) -> ToolRegistry:
        registry = ToolRegistry()
        for tool in self._tools:
            registry.register(tool)
        return registry
