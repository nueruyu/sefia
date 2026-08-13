from typing import Sequence

from typing_extensions import final, override

from .._tool_system import ToolCollector, ToolRegistry
from ..inference import Capability


@final
class CompositeToolCollector(ToolCollector):
    """Composes several collectors into one, merging their registries.

    Name collisions across collectors raise ``ToolConflictError`` (via
    ``ToolRegistry.register``), so introspected and pre-built tools share a
    single namespace.
    """

    def __init__(self, collectors: Sequence[ToolCollector]):
        self._collectors = list(collectors)

    @override
    def collect(self, capabilities: list[Capability]) -> ToolRegistry:
        registry = ToolRegistry()
        for collector in self._collectors:
            for tool in collector.collect(capabilities).get_all():
                registry.register(tool)
        return registry
