import inspect
from typing import Any, Callable, cast

from typing_extensions import final, override

from .._introspection import declared_fields, declared_methods, is_protocol
from .._tool_system import (
    ToolCollector,
    ToolFunctionInspector,
    ToolRegistry,
    bears_tools,
    get_stream_handler,
    is_concurrent,
    role_interface,
)
from ..inference import Capability
from ..streaming import StreamHandler


@final
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
    values never widen the surface (the introspection itself lives in
    ``sefia._introspection``) — and the exposed interface is the declared
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

    @override
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

        if declared is not None and is_protocol(declared):
            # Surface path: the protocol's declarations are the whole grant.
            for method_name, schema_fn in declared_methods(declared).items():
                self._add(registry, cap.value, method_name, schema_fn)
            self._collect_fields(registry, cap.value, declared, gated=False)
        else:
            # Class-body path: only Tools-marked field declarations expose.
            container = declared if declared is not None else type(cap.value)
            self._collect_fields(registry, cap.value, container, gated=True)

    def _collect_fields(
        self, registry: ToolRegistry, holder: object, container: type, *, gated: bool
    ) -> None:
        for field_name, annotation in declared_fields(container).items():
            if gated and not bears_tools(annotation):
                continue
            interface = role_interface(annotation)
            if not inspect.isclass(interface) or interface.__module__ == "builtins":
                continue
            field_value = getattr(holder, field_name, None)
            if field_value is None or field_value is holder:
                continue
            for method_name, schema_fn in declared_methods(interface).items():
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
            concurrent=_resolve_concurrent(bound),
        )


def _resolve_concurrent(bound: Callable[..., Any]) -> bool:
    """Whether the tool's implementation method carries ``@concurrent``.

    Read off ``bound``'s own function, like the ``@preview`` handler — the
    marker describes the concrete implementation, not the declared interface.
    """
    return is_concurrent(getattr(bound, "__func__", bound))


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
        return cast(
            StreamHandler,
            handler.__get__(target_self, cast(type[Any], type(target_self))),
        )
    return handler
