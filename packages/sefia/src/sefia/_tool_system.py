import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

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


def _callable_identity(func: Callable[..., Any]) -> Callable[..., Any]:
    return inspect.unwrap(getattr(func, "__func__", func))


@dataclass(frozen=True)
class ToolDefinition:
    """The LLM-facing definition of a tool.

    ``parameters`` is a JSON Schema. ``to_dict`` renders the exact envelope the
    inference strategy embeds into the prompt, so the model never sees any
    Python- or Pydantic-specific detail.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolFunctionInspector(Protocol):
    """Inspects a tool's Python function.

    A ``SignatureTool`` delegates to this to turn its callable into a neutral
    ``ToolDefinition`` and to bind decoded arguments to the callable's
    parameters. The return values are neutral (a ``ToolDefinition`` and a plain
    ``dict``) — the pydantic implementation never leaks its types across this
    boundary.
    """

    def tool_name(self, func: Callable[..., Any]) -> str:
        """A stable, sanitized tool-call name derived from ``func``."""
        ...

    def definition(self, func: Callable[..., Any], *, name: str) -> ToolDefinition:
        """The tool definition (JSON Schema parameters) reflected from ``func``."""
        ...

    def bind(
        self, func: Callable[..., Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate/coerce decoded ``arguments`` into keyword arguments for ``func``.

        Shape (required/type/extra) is already enforced upstream by the decision
        model; this only coerces values to the callable's declared types and
        passes any additional keys through unchanged.
        """
        ...


class Tool(ABC):
    """A tool the LLM may call.

    Concrete tools differ along two independent axes — where their parameter
    schema comes from (an introspected callable vs. a raw JSON Schema) and how a
    call is executed (a local coroutine, an HTTP round-trip, ...). Both axes are
    expressed as behavior on the tool: ``definition`` produces the LLM-facing
    schema and ``invoke`` runs the call. Nothing implementation-specific (a
    Pydantic type, a validator) is exposed on this surface.
    """

    name: str
    stream_handler: StreamHandler | None

    @abstractmethod
    def definition(self) -> ToolDefinition:
        """The LLM-facing definition embedded into the prompt."""
        ...

    @abstractmethod
    async def invoke(self, arguments: dict[str, Any]) -> Any:
        """Execute the call for decoded ``arguments`` and return its result."""
        ...


class SignatureTool(Tool):
    """A tool whose schema is introspected from a typed Python callable.

    ``function`` is what runs. ``schema_source`` is the callable the schema is
    derived from — normally the same callable, but for a tool discovered through
    a ``Protocol``-narrowed field it is the Protocol's own method (its declared
    signature and docstring), while ``function`` stays the concrete, bound
    implementation.
    """

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        name: str,
        schema_source: Callable[..., Any],
        inspector: ToolFunctionInspector,
        stream_handler: StreamHandler | None = None,
    ):
        self.name = name
        self.stream_handler = stream_handler
        self._function = function
        self._schema_source = schema_source
        self._inspector = inspector

    def definition(self) -> ToolDefinition:
        return self._inspector.definition(self._schema_source, name=self.name)

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        bound = self._inspector.bind(self._function, arguments)
        return await _maybe_await(self._function(**bound))


class JsonSchemaTool(Tool):
    """A tool whose schema is supplied directly as a raw JSON Schema.

    ``handler`` receives the decoded arguments verbatim (``handler(**arguments)``)
    — the JSON stays JSON, which is what a transport-backed handler (e.g. an MCP
    round-trip) expects. There is no signature to introspect.
    """

    def __init__(
        self,
        handler: Callable[..., Any],
        *,
        name: str,
        parameters: dict[str, Any],
        description: str = "",
        stream_handler: StreamHandler | None = None,
    ):
        self.name = name
        self.stream_handler = stream_handler
        self._handler = handler
        self._parameters = parameters
        self._description = description

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self._description,
            parameters=self._parameters,
        )

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        return await _maybe_await(self._handler(**arguments))


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is awaitable, otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


_default_inspector: ToolFunctionInspector | None = None


def _get_default_inspector() -> ToolFunctionInspector:
    global _default_inspector
    if _default_inspector is None:
        from .pydantic._model_backend import PydanticModelBackend

        _default_inspector = PydanticModelBackend()
    return _default_inspector


class ToolRegistry:
    """Stores and provides access to registered tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a pre-built tool. Raises on a name collision."""
        if tool.name in self._tools:
            raise ToolConflictError(
                f"A tool with the name '{tool.name}' already exists."
            )
        self._tools[tool.name] = tool

    def add(
        self,
        func: Callable[..., Any],
        *,
        name: str,
        schema_source: Callable[..., Any] | None = None,
        inspector: ToolFunctionInspector | None = None,
        stream_handler: StreamHandler | None = None,
    ) -> None:
        """Register a ``SignatureTool`` built from a typed callable."""
        self.register(
            SignatureTool(
                func,
                name=name,
                schema_source=schema_source or func,
                inspector=inspector or _get_default_inspector(),
                stream_handler=stream_handler,
            )
        )

    def add_json_tool(
        self,
        handler: Callable[..., Any],
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        stream_handler: StreamHandler | None = None,
    ) -> None:
        """Register a ``JsonSchemaTool`` from a raw JSON Schema and a handler."""
        self.register(
            JsonSchemaTool(
                handler,
                name=name,
                parameters=parameters,
                description=description,
                stream_handler=stream_handler,
            )
        )

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_by_function(self, func: Callable[..., Any]) -> list[Tool]:
        """Return tools whose executable callable matches ``func``."""
        target = _callable_identity(func)
        return [
            tool
            for tool in self._tools.values()
            if _callable_identity(tool.function) is target
        ]

    def get_all(self) -> list[Tool]:
        return list(self._tools.values())

    def get_names(self) -> list[str]:
        return list(self._tools.keys())


class ToolCollector(ABC):
    """Builds a registry of tools for an object."""

    @abstractmethod
    def collect(self, instance: object) -> ToolRegistry: ...


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
