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

    Tools come from the receiver (``self``/``cls``) of the ``@infer`` call, in
    one of two modes:

    * **Unannotated receiver** — the instance's class-level field declarations
      are scanned, and a field is exposed only if annotated with the ``Tools``
      role alias (``_web: Tools[WebToolkit]``). An unmarked or undeclared field
      exposes nothing (fail-closed); the instance's own methods are never
      exposed.
    * **Surface-annotated receiver** — a ``self`` annotated with a ``Protocol``
      replaces the class-body scan with that protocol's declarations, which are
      granted wholesale: its methods (``_``-prefixed included) become tools
      bound to the instance, and its field/property declarations expose their
      declared type's members. Annotating ``self`` is itself the opt-in, so
      the protocol needs no marker and stays a plain interface.

    Either way discovery is a pure function of static declarations — runtime
    values never widen the surface — and the exposed interface is the declared
    type: a concrete class contributes its public methods, a ``Protocol``
    exactly its declared members.

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

        declared = role_interface(cap.declared) if cap.declared is not None else None
        if declared is not None and not inspect.isclass(declared):
            return

        if declared is not None and _is_protocol(declared):
            # Surface path: the protocol's declarations are the whole grant.
            for method_name, schema_fn in _members(declared).items():
                self._add(registry, cap.value, method_name, schema_fn)
            self._collect_fields(registry, cap.value, declared, gated=False)
        else:
            # Class-body path: only Tools-marked field declarations expose.
            container = declared if declared is not None else type(cap.value)
            self._collect_fields(registry, cap.value, container, gated=True)

    def _collect_fields(
        self, registry: ToolRegistry, holder: object, container: type, *, gated: bool
    ) -> None:
        for field_name, annotation in _fields(container).items():
            if gated and not bears_tools(annotation):
                continue
            interface = role_interface(annotation)
            if not inspect.isclass(interface) or interface.__module__ == "builtins":
                continue
            field_value = getattr(holder, field_name, None)
            if field_value is None or field_value is holder:
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

    Scans ``__mro__`` via each class's own ``vars`` — never ``getattr`` on an
    instance, so a ``@property`` getter's side effects are never triggered;
    non-function descriptors are excluded. A concrete class exposes only public
    methods; a ``Protocol`` also exposes its ``_``-prefixed declared members,
    since a protocol is an explicit allowlist.
    """
    is_proto = _is_protocol(cls)
    methods: dict[str, Callable[..., Any]] = {}
    for base in cls.__mro__:
        # Skips object, Protocol, and typing machinery.
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

    Class-level annotations, plus read-only ``property`` declarations whose
    return type is the field's interface (the form that lets a surface protocol
    re-narrow a field — a plain protocol attribute is invariant). Resolution is
    per field and fail-closed: an unresolvable annotation is skipped with a
    debug log, never widened to the runtime type.
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
