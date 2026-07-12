import inspect
import logging
import sys
from typing import Any, Callable

from .._tool_system import (
    Capability,
    ToolCollector,
    ToolFunctionInspector,
    ToolRegistry,
    bears_tools,
    get_stream_handler,
    role_interface,
)
from ..streaming import StreamHandler

_log = logging.getLogger("sefia.tools")


class DefaultToolCollector(ToolCollector):
    """The default implementation of ToolCollector.

    Tools are gated by the ``Tools`` role marker — there is no ambient
    authority. A member becomes a tool only when it is reachable through an
    unbroken chain of ``Tools``-bearing **declared** types, starting at a
    capability parameter:

    * The instance's own methods are not tools unless a role-bearing surface
      (a ``self: SomeTools`` annotation) declares them; a plain service class
      does not bear ``Tools``, so a run never recurses into itself.
    * Each held field is exposed only if its **class-level declared type** bears
      ``Tools`` — a field with no class-level annotation, or one whose declared
      type is not a toolkit, exposes nothing (fail-closed). Runtime values never
      widen the surface: discovery is a pure function of static declarations.
    * A field's exposed interface is its declared type (``role_interface``): a
      concrete class exposes its public methods, a ``Protocol`` exposes exactly
      its declared members (including ``_``-prefixed ones — an explicit
      allowlist, e.g. opting a private method in as a tool).

    The collector records neutral tool metadata; strategy-specific schema
    generation happens later, from each tool's ``schema_source``.
    """

    def __init__(self, inspector: ToolFunctionInspector | None = None):
        if inspector is None:
            from ..pydantic._model_backend import PydanticModelBackend

            inspector = PydanticModelBackend()
        self._inspector = inspector

    def collect(self, capabilities: list[Capability]) -> ToolRegistry:
        registry = ToolRegistry()
        for cap in capabilities:
            self._collect_capability(registry, cap)
        return registry

    def _collect_capability(self, registry: ToolRegistry, cap: Capability) -> None:
        if cap.value is None:
            return

        # The container whose held fields are scanned. For the self/cls
        # convention (no declared type) it is the value's own class; otherwise
        # it is the role-bearing declared type (surface protocol or toolkit).
        if cap.declared is None:
            container = type(cap.value)
            expose_own = False  # a bare service is a container, not a toolkit
        else:
            container = role_interface(cap.declared)
            expose_own = bears_tools(cap.declared)

        if not inspect.isclass(container):
            return

        # Tier 0 — the capability's own type is itself a tool surface: its
        # members bind to the value (a directly-passed toolkit, or a surface
        # protocol opting the instance's own methods in).
        if expose_own:
            for method_name, schema_fn in _members(container).items():
                self._add(registry, cap.value, method_name, schema_fn)

        # Tier 1 — held fields whose declared type bears ``Tools``.
        for field_name, field_type in _fields(container).items():
            if not bears_tools(field_type):
                continue
            field_value = getattr(cap.value, field_name, None)
            if field_value is None or field_value is cap.value:
                continue
            interface = role_interface(field_type)
            if not inspect.isclass(interface):
                continue
            for method_name, schema_fn in _members(interface).items():
                self._add(registry, field_value, method_name, schema_fn)

    def _add(
        self,
        registry: ToolRegistry,
        holder: object,
        method_name: str,
        schema_fn: Callable[..., Any],
    ) -> None:
        bound = getattr(holder, method_name, None)
        if not callable(bound):
            return
        registry.add(
            bound,
            name=self._inspector.tool_name(schema_fn),
            schema_source=schema_fn,
            inspector=self._inspector,
            stream_handler=_resolve_stream_handler(bound),
        )


def _is_protocol(cls: type) -> bool:
    return bool(getattr(cls, "_is_protocol", False))


def _members(cls: type) -> dict[str, Callable[..., Any]]:
    """The tool methods a class or ``Protocol`` exposes, by name.

    Scans ``__mro__`` and each class's own ``vars`` (never ``dir()`` +
    ``getattr`` on an instance, so a ``@property`` getter's side effects are
    never triggered). Properties and other non-function descriptors are excluded
    — a tool is a callable method.

    For a **concrete class** only public (non ``_``-prefixed) methods are
    exposed. For a **Protocol** every declared member is exposed, including
    ``_``-prefixed ones: the protocol is an explicit allowlist authored for this
    surface, so a declared private method is a deliberate opt-in.
    """
    is_proto = _is_protocol(cls)
    methods: dict[str, Callable[..., Any]] = {}
    for base in cls.__mro__:
        # Skips object, Generic, Protocol, the role markers, and typing
        # machinery — leaving only user-authored classes/protocols.
        if base.__module__ in ("builtins", "typing"):
            continue
        for name, raw in vars(base).items():
            if name in methods or name.startswith("__"):
                continue
            if name.startswith("_") and not is_proto:
                continue
            if isinstance(raw, (staticmethod, classmethod)):
                methods[name] = raw.__func__
            elif inspect.isfunction(raw):
                methods[name] = raw
    return methods


def _fields(cls: type) -> dict[str, Any]:
    """The declared held-field types of ``cls``, by attribute name.

    Combines class-level annotations with read-only ``property`` declarations
    (whose return type is the field's interface — the form that lets a surface
    protocol re-narrow a field, since a plain protocol attribute is invariant).
    Resolution is **per field and fail-closed**: an annotation that cannot be
    resolved (a forward reference, a ``TYPE_CHECKING``-only name) is skipped with
    a debug log rather than widening or crashing discovery.
    """
    fields: dict[str, Any] = {}
    for base in cls.__mro__:
        if base.__module__ in ("builtins", "typing"):
            continue
        globalns = getattr(sys.modules.get(base.__module__, None), "__dict__", {})
        localns = dict(vars(base))
        raw = base.__dict__.get("__annotations__", {})
        for name, annotation in raw.items():
            if name in fields:
                continue
            resolved = _resolve(annotation, globalns, localns)
            if resolved is not None:
                fields[name] = resolved
        for name, member in vars(base).items():
            if name in fields or not isinstance(member, property) or member.fget is None:
                continue
            try:
                ret = inspect.get_annotations(member.fget, eval_str=True).get("return")
            except Exception:  # noqa: BLE001 — fail-closed on any resolution error
                ret = None
            if ret is not None:
                fields[name] = ret
    return fields


def _resolve(annotation: Any, globalns: dict, localns: dict) -> Any:
    if not isinstance(annotation, str):
        return annotation
    try:
        return eval(annotation, globalns, localns)  # noqa: S307 — resolving a type hint
    except Exception:  # noqa: BLE001 — fail-closed: an unresolved field is not discovered
        _log.debug("skipping unresolvable field annotation %r", annotation)
        return None


def _resolve_stream_handler(bound: Callable[..., Any]) -> StreamHandler | None:
    """The ``@preview`` handler bound to a tool's target, or ``None``.

    ``preview`` attaches the handler to the tool's *implementation* method, so
    it is looked up on ``bound``'s own underlying function — never on the
    declared interface's method, which can be a different object under
    ``Protocol`` narrowing. ``bound`` carries ``__self__`` for an instance or
    class method, so the handler is bound to that same target; a ``staticmethod``
    tool has no ``__self__`` and the handler is returned unbound.
    """
    target_self = getattr(bound, "__self__", None)
    implementation = getattr(bound, "__func__", bound)
    handler = get_stream_handler(implementation)
    if handler is None:
        return None
    if target_self is not None:
        return handler.__get__(target_self, type(target_self))
    return handler
