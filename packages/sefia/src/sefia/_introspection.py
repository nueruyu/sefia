"""Annotation and class-shape introspection.

Everything that *reads* Python's static declarations for tool discovery lives
here: the ``Tools`` role alias and its resolver, and the scanners that turn a
class or ``Protocol`` into its declared methods and held-field types. The
collector composes these; it contains no introspection of its own.
"""

import inspect
import logging
import sys
import types
from typing import Annotated, Any, Callable, TypeVar, Union, get_args, get_origin

from typing_extensions import TypeAliasType

_log = logging.getLogger("sefia.tools")


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


def role_interface(annotation: Any) -> Any:
    """The declared interface under ``annotation``'s role/``Optional`` wrappers."""
    return unwrap_role(annotation)[1]


def is_protocol(cls: type) -> bool:
    """Whether ``cls`` is a ``typing.Protocol`` (as opposed to a concrete class)."""
    return bool(getattr(cls, "_is_protocol", False))


def exposed_methods(cls: type) -> dict[str, Callable[..., Any]]:
    """The tool methods a class or ``Protocol`` exposes, by name.

    Scans ``__mro__`` via each class's own ``vars`` — never ``getattr`` on an
    instance, so a ``@property`` getter's side effects are never triggered;
    non-function descriptors are excluded. A concrete class exposes only public
    methods; a ``Protocol`` also exposes its ``_``-prefixed declared members,
    since a protocol is an explicit allowlist.
    """
    is_proto = is_protocol(cls)
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


def declared_fields(cls: type) -> dict[str, Any]:
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
