from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from .exceptions import ToolConflictError
from .streaming import StreamHandler


@dataclass(frozen=True)
class Tool:
    """Represents a callable tool registered for inference.

    ``function`` is what gets called. ``schema_function`` is what the schema
    (name, docstring, parameters) is built from — normally the same callable,
    but for a tool discovered through a ``Protocol``-narrowed field, it is the
    Protocol's own method (its declared signature and docstring), while
    ``function`` stays the concrete, bound implementation that actually runs.
    """

    name: str
    function: Callable[..., Any]
    schema_function: Callable[..., Any] | None = None
    stream_handler: StreamHandler | None = None

    @property
    def schema(self) -> Callable[..., Any]:
        """The callable to build the tool's schema from; falls back to ``function``."""
        return self.schema_function or self.function


class ToolRegistry:
    """Stores and provides access to registered tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def add(
        self,
        func: Callable[..., Any],
        *,
        name: str,
        schema_function: Callable[..., Any] | None = None,
        stream_handler: StreamHandler | None = None,
    ) -> None:
        if name in self._tools:
            raise ToolConflictError(f"A tool with the name '{name}' already exists.")
        self._tools[name] = Tool(
            name=name,
            function=func,
            schema_function=schema_function,
            stream_handler=stream_handler,
        )

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
