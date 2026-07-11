import inspect
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from .._tool_system import (
    ToolCollector,
    ToolFunctionInspector,
    ToolRegistry,
    get_stream_handler,
    is_concurrent,
)
from ..streaming import StreamHandler

# Annotations that declare no usable interface; a field so annotated falls
# back to its runtime type instead of resolving to a type with no methods.
_NO_INTERFACE = (Any, object)


class DefaultToolCollector(ToolCollector):
    """
    The default implementation of ToolCollector.

    Tools are the public surface of what the instance holds — plain Python
    visibility, no marker or registry:

    * The instance's own methods are never tools (no self-recursion; hold
      another service as a dependency to use it as a tool).
    * Each dependency the instance holds in an attribute (public or private)
      contributes its public (non ``_``-prefixed) methods.
    * A field's exposed interface is its **class-level annotation** when one is
      declared (a concrete class exposes its public methods; a ``Protocol``
      exposes only its declared members) — otherwise it falls back to the
      **runtime value's concrete type**. ``__init__`` parameter annotations are
      never consulted: the mapping from a parameter to the attribute it is
      stored under is arbitrary runtime code and cannot be recovered.

    The collector records neutral tool metadata. Strategy-specific schema
    generation happens later in the inference strategy, from each tool's
    ``schema_source`` (the interface's method — the class-level annotation's
    method, or the runtime type's, whichever supplied the surface).
    """

    def __init__(self, inspector: ToolFunctionInspector | None = None):
        if inspector is None:
            from ..pydantic._model_backend import PydanticModelBackend

            inspector = PydanticModelBackend()
        self._inspector = inspector

    def collect(self, instance: object) -> ToolRegistry:
        registry = ToolRegistry()
        hints = _class_hints(type(instance))

        for attr_name in _held_attr_names(instance):
            value = getattr(instance, attr_name, None)
            if value is None or value is instance:
                continue

            interface = _resolve_interface(hints.get(attr_name), value)
            if interface is None:
                continue

            for method_name, schema_fn in _public_methods(interface).items():
                bound = getattr(value, method_name, None)
                if not callable(bound):
                    continue
                registry.add(
                    bound,
                    name=self._inspector.tool_name(schema_fn),
                    schema_source=schema_fn,
                    inspector=self._inspector,
                    stream_handler=_resolve_stream_handler(bound),
                    concurrent=_resolve_concurrent(bound),
                )

        return registry


def _held_attr_names(instance: object) -> set[str]:
    # Slotted classes have no __dict__, so also gather names from __slots__
    # across the class hierarchy.
    attr_names = set(getattr(instance, "__dict__", {}))
    for cls in type(instance).__mro__:
        slots = cls.__dict__.get("__slots__", None)
        if not slots:
            continue
        if isinstance(slots, str):
            attr_names.add(slots)
        else:
            attr_names.update(slots)
    return attr_names


def _class_hints(cls: type) -> dict[str, Any]:
    try:
        # include_extras=False (the default) also resolves Annotated[X, ...]
        # down to X, since nothing here reads the extra metadata.
        return get_type_hints(cls)
    except NameError:
        # An unresolvable forward reference (e.g. a name only available under
        # TYPE_CHECKING) must not break discovery; those fields simply fall
        # back to their runtime type below. Any other exception is a genuine
        # bug in the annotation and should surface, not be swallowed.
        return {}


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _resolve_interface(annotation: Any, value: object) -> type | None:
    """The type whose public methods are exposed for this held field."""
    declared = _unwrap_optional(annotation) if annotation is not None else None
    if declared in _NO_INTERFACE:
        declared = None
    interface = declared if inspect.isclass(declared) else type(value)
    # A held primitive/builtin instance (str, list, int, ...) must not leak its
    # methods as tools; user-defined and third-party classes are fair game.
    if interface.__module__ == "builtins":
        return None
    return interface


def _public_methods(cls: type) -> dict[str, Callable[..., Any]]:
    """The class's public methods (own + inherited), by name.

    Scans ``__mro__`` and each class's own ``__dict__`` rather than ``dir()`` +
    ``getattr`` on the *instance*, so accessing a member never triggers a
    third-party object's ``@property`` getters or other side effects.
    Properties and other non-function descriptors are excluded — a tool must be
    a callable method, not an attribute. ``cls`` may be a concrete class or a
    ``Protocol``; either way this returns only what it declares or inherits,
    which is exactly the exposed surface.
    """
    methods: dict[str, Callable[..., Any]] = {}
    for base in cls.__mro__:
        for name, raw in vars(base).items():
            if name in methods or name.startswith("_"):
                continue
            if isinstance(raw, (staticmethod, classmethod)):
                methods[name] = raw.__func__
            elif inspect.isfunction(raw):
                methods[name] = raw
    return methods


def _resolve_concurrent(bound: Callable[..., Any]) -> bool:
    """Whether the tool's *implementation* method carries ``@concurrent``.

    Like the ``@preview`` handler, the marker describes a runtime property of
    the concrete implementation, so it is read off ``bound``'s own underlying
    function — never off the declared interface's method, which can be a
    different object under ``Protocol`` narrowing.
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
        return handler.__get__(target_self, type(target_self))
    return handler
