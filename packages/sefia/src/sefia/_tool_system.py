import inspect
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    Protocol,
    Union,
    get_args,
    get_origin,
)

from .exceptions import ToolConflictError
from .streaming import StreamHandler


class Tools(Protocol):
    """Role marker: a type whose members the model may call.

    Declared by inheritance (``class WebToolkit(Tools)``; for a narrowing
    protocol, re-inherit: ``class ReadOnlyWeb(Tools, Protocol)``), or at a use
    site for types you cannot edit (``Annotated[VendorClient, Tools]``).
    Discovery only exposes fields/parameters whose declared type bears this
    role — holding an object is not enough. Membership is nominal (MRO or
    ``Annotated`` metadata), not structural.
    """


class Context(Protocol):
    """Role marker: a type whose members are readable by the model.

    Reserved for rendering declared data members into the prompt; detected
    today, rendered in a later stage.
    """


def _strip_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Split ``Annotated[T, *meta]`` into ``(T, meta)``; else ``(annotation, ())``."""
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        return args[0], args[1:]
    return annotation, ()


def _strip_optional(annotation: Any) -> Any:
    """Unwrap ``T | None`` (and ``Optional[T]``) to ``T``; leave others unchanged."""
    if get_origin(annotation) in (Union, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _bears_marker(annotation: Any, marker: type) -> bool:
    inner, meta = _strip_annotated(annotation)
    if marker in meta:
        return True
    inner = _strip_optional(inner)
    inner, meta = _strip_annotated(inner)
    if marker in meta:
        return True
    return isinstance(inner, type) and marker in getattr(inner, "__mro__", ())


def bears_tools(annotation: Any) -> bool:
    """Whether ``annotation``'s declared type carries the ``Tools`` role."""
    return _bears_marker(annotation, Tools)


def bears_context(annotation: Any) -> bool:
    """Whether ``annotation``'s declared type carries the ``Context`` role."""
    return _bears_marker(annotation, Context)


def bears_role(annotation: Any) -> bool:
    """Whether ``annotation`` carries either role marker."""
    return bears_tools(annotation) or bears_context(annotation)


def role_interface(annotation: Any) -> Any:
    """The class/``Protocol`` carrying the role (``Annotated``/``Optional`` stripped)."""
    inner, _ = _strip_annotated(annotation)
    inner = _strip_optional(inner)
    inner, _ = _strip_annotated(inner)
    return inner


# ``self``/``cls`` are capability parameters by convention, without a role annotation.
_RECEIVER_NAMES = ("self", "cls")


@dataclass(frozen=True)
class Capability:
    """A capability parameter's runtime value and its declared type.

    ``declared`` is ``None`` for the ``self``/``cls`` convention — the collector
    then scans the value's own class for role-bearing held fields.
    """

    value: object
    declared: Any


def is_capability_parameter(name: str, declared: Any) -> bool:
    """Whether a parameter carries tools rather than task data."""
    if name in _RECEIVER_NAMES:
        return True
    return declared is not None and bears_role(declared)


def capability_names(
    bound_arguments: dict[str, Any], type_hints: dict[str, Any]
) -> set[str]:
    """The names of the capability parameters among ``bound_arguments``."""
    return {
        name
        for name in bound_arguments
        if is_capability_parameter(name, type_hints.get(name))
    }


def capabilities(
    bound_arguments: dict[str, Any], type_hints: dict[str, Any]
) -> list[Capability]:
    """Extract the capability parameters (value + declared type) from a call."""
    out: list[Capability] = []
    for name, value in bound_arguments.items():
        declared = type_hints.get(name)
        if is_capability_parameter(name, declared):
            out.append(Capability(value=value, declared=declared))
    return out

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


class ToolFunctionInspector(ABC):
    """Inspects a tool's Python function.

    A ``SignatureTool`` delegates to this to turn its callable into a neutral
    ``ToolDefinition`` and to bind decoded arguments to the callable's
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

    @property
    def function(self) -> Callable[..., Any] | None:
        """The local Python callable this tool executes, if any.

        Used by function-based lookups (``ToolRegistry.get_by_function``).
        ``None`` for tools executed over a transport with no local callable.
        """
        return None


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

    @property
    def function(self) -> Callable[..., Any]:
        return self._function


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
        if inspector is None:
            from .pydantic._model_backend import PydanticModelBackend

            inspector = PydanticModelBackend()
        self.register(
            SignatureTool(
                func,
                name=name,
                schema_source=schema_source or func,
                inspector=inspector,
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
            if tool.function is not None and _callable_identity(tool.function) is target
        ]

    def get_all(self) -> list[Tool]:
        return list(self._tools.values())

    def get_names(self) -> list[str]:
        return list(self._tools.keys())


class ToolCollector(ABC):
    """Builds a registry of tools from a call's capability parameters."""

    @abstractmethod
    def collect(self, capabilities: list[Capability]) -> ToolRegistry: ...
