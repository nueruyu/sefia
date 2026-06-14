from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from .exceptions import ToolConflictError


@dataclass(frozen=True)
class Tool:
    """Represents a callable tool with its schema."""

    function: Callable[..., Any]
    schema: dict[str, Any]


class ToolRegistry:
    """Stores and provides access to registered tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def add(self, func: Callable[..., Any], schema: dict[str, Any]) -> None:
        tool_name = schema["function"]["name"]
        if tool_name in self._tools:
            raise ToolConflictError(
                f"A tool with the name '{tool_name}' already exists."
            )
        self._tools[tool_name] = Tool(function=func, schema=schema)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        return list(self._tools.values())

    def get_names(self) -> list[str]:
        return list(self._tools.keys())


class ToolCollector(ABC):
    """Builds a registry of tools for an object."""

    @abstractmethod
    def collect(self, instance: object) -> ToolRegistry: ...
