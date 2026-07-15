import inspect
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from typing_extensions import TypeAliasType

from .exceptions import ToolConflictError
from .streaming import StreamHandler


class _RoleMarker:
    """Sentinel a role alias plants in ``Annotated`` metadata."""

    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return f"sefia.{self._name}"


_TOOLS = _RoleMarker("Tools")
_CONTEXT = _RoleMarker("Context")

T = TypeVar("T")

Tools = TypeAliasType("Tools", Annotated[T, _TOOLS], type_params=(T,))
"""Role alias: a field whose members the model may call.

Written in a class-level field annotation — ``_web: Tools[WebToolkit]``,
narrowed with ``Tools[ReadOnlyWeb]`` — it grants that one field. The wrapped
type stays a plain class/``Protocol`` (checkers treat ``Tools[T]`` as ``T``),
and discovery exposes only fields so annotated: holding an object is not
enough.
"""

Context = TypeAliasType("Context", Annotated[T, _CONTEXT], type_params=(T,))
"""Role alias: a field whose value the model may read.

Reserved for rendering declared data members into the prompt; detected today,
rendered in a later stage.
"""


def unwrap_role(annotation: Any) -> tuple[frozenset, Any]:
    """Resolve ``annotation`` to ``(role markers, interface)``.

    Peels ``Annotated`` layers, role aliases (bare or subscripted), and
    ``Optional`` — in any nesting order — collecting markers along the way.
    The remainder is the declared interface.
    """
    markers: set[_RoleMarker] = set()
    hint = annotation
    for _ in range(16):  # annotations are shallow; bound guards against cycles
        origin = get_origin(hint)
        if origin is Annotated:
            args = get_args(hint)
            markers.update(a for a in args[1:] if isinstance(a, _RoleMarker))
            hint = args[0]
        elif getattr(origin, "__value__", None) is not None:
            # A subscripted alias: markers live in its body; the type argument
            # is the interface (our aliases are ``Annotated[T, marker]``).
            value = origin.__value__
            if get_origin(value) is Annotated:
                markers.update(
                    a for a in get_args(value)[1:] if isinstance(a, _RoleMarker)
                )
                hint = get_args(hint)[0]
            else:
                hint = value
        elif getattr(hint, "__value__", None) is not None:
            hint = hint.__value__  # a bare (unsubscripted) alias
        elif origin in (Union, types.UnionType):
            non_none = [a for a in get_args(hint) if a is not type(None)]
            if len(non_none) != 1:
                break
            hint = non_none[0]
        else:
            break
    return frozenset(markers), hint


def bears_tools(annotation: Any) -> bool:
    """Whether ``annotation`` carries the ``Tools`` role."""
    return _TOOLS in unwrap_role(annotation)[0]


def bears_context(annotation: Any) -> bool:
    """Whether ``annotation`` carries the ``Context`` role."""
    return _CONTEXT in unwrap_role(annotation)[0]


def role_interface(annotation: Any) -> Any:
    """The declared interface under ``annotation``'s role/``Optional`` wrappers."""
    return unwrap_role(annotation)[1]


# Only ``self``/``cls`` carry tools — by convention, with no marker. Tool
# dependencies are expressed through classes; plain-function parameters are
# always task data.
_RECEIVER_NAMES = ("self", "cls")


@dataclass(frozen=True)
class Capability:
    """An ``@infer`` call's receiver and its declared type.

    ``declared`` is the receiver's annotation when present (a surface
    ``Protocol`` selecting this method's tools), else ``None``.
    """

    value: object
    declared: Any


def capability_names(bound_arguments: dict[str, Any]) -> set[str]:
    """The receiver names among ``bound_arguments``."""
    return {name for name in bound_arguments if name in _RECEIVER_NAMES}


def capabilities(
    bound_arguments: dict[str, Any], type_hints: dict[str, Any]
) -> list[Capability]:
    """Extract the capability parameters (receiver + surface type) from a call."""
    return [
        Capability(value=value, declared=type_hints.get(name))
        for name, value in bound_arguments.items()
        if name in _RECEIVER_NAMES
    ]

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

    def remove(self, name: str) -> None:
        """Drop a tool by name; a missing name is a no-op."""
        self._tools.pop(name, None)

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
