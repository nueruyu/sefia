from typing import Sequence

from .._tool_system import ToolCollector, ToolRegistry


class CompositeToolCollector(ToolCollector):
    """Composes several collectors into one, merging their registries.

    Name collisions across collectors raise ``ToolConflictError`` (via
    ``ToolRegistry.register``), so introspected and pre-built tools share a
    single namespace.
    """

    def __init__(self, collectors: Sequence[ToolCollector]):
        self._collectors = list(collectors)

    def collect(self, instance: object) -> ToolRegistry:
        registry = ToolRegistry()
        for collector in self._collectors:
            for tool in collector.collect(instance).get_all():
                registry.register(tool)
        return registry
