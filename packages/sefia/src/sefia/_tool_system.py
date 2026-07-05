from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from .exceptions import ToolConflictError
from .streaming import StreamHandler

# The attribute under which a tool method carries its ``@preview`` stream
# handler. Kept private to this module; ``preview`` and the collector go
# through the accessors below so the raw name never crosses a module boundary.
_STREAM_HANDLER_ATTR = "__sefia_stream_handler__"


def set_stream_handler(func: Callable[..., Any], handler: StreamHandler) -> None:
    """Attach ``handler`` to ``func`` as its tool's argument-stream preview."""
    setattr(func, _STREAM_HANDLER_ATTR, handler)


def get_stream_handler(func: Callable[..., Any]) -> StreamHandler | None:
    """Return the stream handler attached to ``func``, or ``None``."""
    return getattr(func, _STREAM_HANDLER_ATTR, None)


@dataclass(frozen=True)
class Tool:
    """Represents a callable tool registered for inference.

    ``function`` is what gets called. ``schema_source`` is the callable the
    tool's schema (name, docstring, parameters) is derived from — normally the
    same callable, but for a tool discovered through a ``Protocol``-narrowed
    field it is the Protocol's own method (its declared signature and
    docstring), while ``function`` stays the concrete, bound implementation
    that actually runs.
    """

    name: str
    function: Callable[..., Any]
    schema_source: Callable[..., Any]
    stream_handler: StreamHandler | None = None


class ToolRegistry:
    """Stores and provides access to registered tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def add(
        self,
        func: Callable[..., Any],
        *,
        name: str,
        schema_source: Callable[..., Any] | None = None,
        stream_handler: StreamHandler | None = None,
    ) -> None:
        if name in self._tools:
            raise ToolConflictError(f"A tool with the name '{name}' already exists.")
        self._tools[name] = Tool(
            name=name,
            function=func,
            schema_source=schema_source or func,
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
