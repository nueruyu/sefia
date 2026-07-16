import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Any, Callable, TypeVar

from typing_extensions import TypeAliasType

from ._introspection import unwrap_annotation
from .exceptions import ToolConflictError
from .inference import Capability
from .streaming import StreamHandler


class _RoleMarker:
    """Sentinel a role alias plants in ``Annotated`` metadata."""

    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return f"sefia.{self._name}"


_TOOLS = _RoleMarker("Tools")

T = TypeVar("T")

Tools = TypeAliasType("Tools", Annotated[T, _TOOLS], type_params=(T,))
"""Role alias: a field whose members the model may call.

Written in a class-level field annotation — ``_web: Tools[WebToolkit]``,
narrowed with ``Tools[ReadOnlyWeb]`` — it grants that one field. The wrapped
type stays a plain class/``Protocol`` (checkers treat ``Tools[T]`` as ``T``),
and discovery exposes only fields so annotated: holding an object is not
enough.
"""


def bears_tools(annotation: Any) -> bool:
    """Whether ``annotation`` carries the ``Tools`` role."""
    metadata, _ = unwrap_annotation(annotation)
    return any(item is _TOOLS for item in metadata)


def role_interface(annotation: Any) -> Any:
    """The declared interface under ``annotation``'s role/``Optional`` wrappers."""
    return unwrap_annotation(annotation)[1]


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


# The attribute under which a tool method carries its ``@concurrent`` marker.
# Same pattern as the stream handler above: the decorator and the collector go
# through the accessors, the raw name stays private to this module.
_CONCURRENT_ATTR = "__sefia_concurrent__"


def set_concurrent(func: Callable[..., Any]) -> None:
    """Mark ``func``'s tool as safe to overlap with other concurrent calls."""
    setattr(func, _CONCURRENT_ATTR, True)


def is_concurrent(func: Callable[..., Any]) -> bool:
    """Whether ``func`` carries the ``@concurrent`` marker."""
    return getattr(func, _CONCURRENT_ATTR, False)


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


class ToolFunctionInspector(ABC):
    """Inspects a tool's Python function.

    A ``SignatureToolEntry`` delegates to this to turn its callable into a
    neutral ``ToolDefinition`` and to bind decoded arguments to the callable's
    parameters. The return values are neutral (a ``ToolDefinition`` and a plain
    ``dict``) — the pydantic implementation never leaks its types across this
    boundary.
    """

    @abstractmethod
    def tool_name(self, func: Callable[..., Any]) -> str:
        """A stable, sanitized tool-call name derived from ``func``."""
        ...

    @abstractmethod
    def definition(self, func: Callable[..., Any], *, name: str) -> ToolDefinition:
        """The tool definition (JSON Schema parameters) reflected from ``func``."""
        ...

    @abstractmethod
    def bind(
        self, func: Callable[..., Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate/coerce decoded ``arguments`` into keyword arguments for ``func``.

        Shape (required/type/extra) is already enforced upstream by the decision
        model; this only coerces values to the callable's declared types and
        passes any additional keys through unchanged.
        """
        ...


class ToolEntry(ABC):
    """A runtime registration record binding a tool's name, schema, and call.

    This is the entry a ``ToolRegistry`` holds — what other frameworks call a
    "Tool object"; in sefia the tool itself is the granted method. Concrete
    entries differ along two independent axes — where their parameter schema
    comes from (an introspected callable vs. a raw JSON Schema) and how a call is
    executed (a local coroutine, an HTTP round-trip, ...). Both axes are
    expressed as behavior on the entry: ``definition`` produces the LLM-facing
    schema and ``invoke`` runs the call. Nothing implementation-specific (a
    Pydantic type, a validator) is exposed on this surface.
    """

    name: str
    stream_handler: StreamHandler | None
    # Whether calls may overlap with other concurrent-marked calls in a batch.
    concurrent: bool = False

    @abstractmethod
    def definition(self) -> ToolDefinition:
        """The LLM-facing definition embedded into the prompt."""
        ...

    @abstractmethod
    async def invoke(self, arguments: dict[str, Any]) -> Any:
        """Execute the call for decoded ``arguments`` and return its result."""
        ...

    @property
    def function(self) -> Callable[..., Any] | None:
        """The local Python callable this tool executes, if any.

        Used by function-based lookups (``ToolRegistry.get_by_function``).
        ``None`` for tools executed over a transport with no local callable.
        """
        return None


class SignatureToolEntry(ToolEntry):
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
        concurrent: bool = False,
    ):
        self.name = name
        self.stream_handler = stream_handler
        self.concurrent = concurrent
        self._function = function
        self._schema_source = schema_source
        self._inspector = inspector

    def definition(self) -> ToolDefinition:
        return self._inspector.definition(self._schema_source, name=self.name)

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        bound = self._inspector.bind(self._function, arguments)
        return await _maybe_await(self._function(**bound))

    @property
    def function(self) -> Callable[..., Any]:
        return self._function


class JsonSchemaToolEntry(ToolEntry):
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
        concurrent: bool = False,
    ):
        self.name = name
        self.stream_handler = stream_handler
        self.concurrent = concurrent
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

    @property
    def function(self) -> Callable[..., Any]:
        return self._handler


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is awaitable, otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


class ToolRegistry:
    """Stores and provides access to registered tools."""

    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}

    def register(self, tool: ToolEntry) -> None:
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
        concurrent: bool = False,
    ) -> None:
        """Register a ``SignatureToolEntry`` built from a typed callable."""
        if inspector is None:
            from .pydantic._model_backend import PydanticModelBackend

            inspector = PydanticModelBackend()
        self.register(
            SignatureToolEntry(
                func,
                name=name,
                schema_source=schema_source or func,
                inspector=inspector,
                stream_handler=stream_handler,
                concurrent=concurrent,
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
        concurrent: bool = False,
    ) -> None:
        """Register a ``JsonSchemaToolEntry`` from a raw JSON Schema and a handler."""
        self.register(
            JsonSchemaToolEntry(
                handler,
                name=name,
                parameters=parameters,
                description=description,
                stream_handler=stream_handler,
                concurrent=concurrent,
            )
        )

    def get(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)

    def get_by_function(self, func: Callable[..., Any]) -> list[ToolEntry]:
        """Return tools whose executable callable matches ``func``."""
        target = _callable_identity(func)
        return [
            tool
            for tool in self._tools.values()
            if tool.function is not None and _callable_identity(tool.function) is target
        ]

    def get_all(self) -> list[ToolEntry]:
        return list(self._tools.values())

    def get_names(self) -> list[str]:
        return list(self._tools.keys())


class ToolCollector(ABC):
    """Builds a registry of tools from a call's capability parameters."""

    @abstractmethod
    def collect(self, capabilities: list[Capability]) -> ToolRegistry: ...
