"""Pure annotation and class-shape reflection.

Nothing in this module knows about sefia's tool concepts: it reads Python's
static declarations — annotation wrappers, class/``Protocol`` members, and
class-level field types — and hands back neutral data. The role vocabulary
built on top of it (the ``Tools`` alias and its predicates) lives in
``_tool_system``; the discovery policy lives in the collector.
"""

import inspect
import logging
import sys
import types
from typing import Annotated, Any, Callable, Union, get_args, get_origin

_log = logging.getLogger(__name__)


def unwrap_annotation(annotation: Any) -> tuple[tuple[Any, ...], Any]:
    """Resolve ``annotation`` to ``(metadata, inner type)``.

    Peels ``Annotated`` layers, type aliases (bare or subscripted), and
    ``Optional`` — in any nesting order — collecting every piece of
    ``Annotated`` metadata along the way. The remainder is the declared type.
    An ambiguous union (more than one non-``None`` arm) stops resolution and is
    returned as-is.
    """
    metadata: list[Any] = []
    hint = annotation
    for _ in range(16):  # annotations are shallow; bound guards against cycles
        origin = get_origin(hint)
        if origin is Annotated:
            args = get_args(hint)
            metadata.extend(args[1:])
            hint = args[0]
        elif getattr(origin, "__value__", None) is not None:
            # A subscripted alias: metadata lives in its body; the type
            # argument substitutes the body's type parameter.
            value = origin.__value__
            if get_origin(value) is Annotated:
                metadata.extend(get_args(value)[1:])
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
    return tuple(metadata), hint


def is_protocol(cls: type) -> bool:
    """Whether ``cls`` is a ``typing.Protocol`` (as opposed to a concrete class)."""
    return bool(getattr(cls, "_is_protocol", False))


def declared_methods(cls: type) -> dict[str, Callable[..., Any]]:
    """The methods a class or ``Protocol`` declares, by name.

    Scans ``__mro__`` via each class's own ``vars`` — never ``getattr`` on an
    instance, so a ``@property`` getter's side effects are never triggered;
    non-function descriptors are excluded. A concrete class contributes only
    public methods; a ``Protocol`` also contributes its ``_``-prefixed declared
    members, since a protocol is an explicit allowlist.
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
    """The declared field types of ``cls``, by attribute name.

    Class-level annotations, plus read-only ``property`` declarations whose
    return type is the field's type (the form that lets a ``Protocol``
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
