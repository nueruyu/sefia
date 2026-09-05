"""Reusable pytest contract for ``ToolCollector`` implementations."""

from dataclasses import dataclass, field
from typing import Any

from .._tool_system.registry import ToolCollector
from ..inference import Capability


def _empty_arguments() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class ToolCollectorCase:
    """One collector setup and the tool behavior it is expected to expose."""

    collector: ToolCollector
    capabilities: list[Capability]
    expected_name: str
    arguments: dict[str, Any] = field(default_factory=_empty_arguments)
    expected_result: Any = None


class ToolCollectorContract:
    """Shared discovery and dispatch behavior required by a tool collector."""

    async def test_collects_the_expected_executable_tool(
        self, tool_collector_case: ToolCollectorCase
    ) -> None:
        registry = tool_collector_case.collector.collect(
            tool_collector_case.capabilities
        )

        assert tool_collector_case.expected_name in registry.get_names()
        tool = registry.get(tool_collector_case.expected_name)
        assert tool is not None
        assert await tool.invoke(tool_collector_case.arguments) == (
            tool_collector_case.expected_result
        )


__all__ = ["ToolCollectorCase", "ToolCollectorContract"]
