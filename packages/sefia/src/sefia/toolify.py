import inspect
import types
from typing import Any, Callable


class Toolset:
    """A bundle of tool callables produced by :func:`toolify`."""

    def __init__(self, tools: list[Callable[..., Any]]):
        self.tools = tools


def toolify(*items: object) -> Toolset:
    """
    Expose external objects and/or plain functions as tools without ``@tool``.

    This is the escape hatch for classes you cannot decorate (for example, a
    third-party library client):

    - A routine (function, async function, or bound method) is exposed directly.
    - Any other object is treated as a tool provider: all of its public
      (non ``_``-prefixed) methods become tools.

    The original objects are not modified; the returned :class:`Toolset` is held
    like any other dependency, e.g. ``self._tools = toolify(SomeClient(), my_fn)``.
    """
    tools: list[Callable[..., Any]] = []
    for item in items:
        # A class object exposes only unbound methods, which would fail when
        # called; skip it (callers pass instances, not classes).
        if inspect.isclass(item):
            continue
        # Any callable — function, async function, bound method,
        # functools.partial, or an object with __call__ — is exposed directly
        # as a single tool, so a partial keeps its bound arguments.
        if callable(item):
            tools.append(item)
            continue
        # Skip builtin primitive/container instances (str, list, generators,
        # map/filter objects, ...) so their public methods do not leak in as
        # tools. Modules and other user objects pass through.
        if type(item).__module__ == "builtins" and not isinstance(
            item, types.ModuleType
        ):
            continue
        # Otherwise treat the object as a tool provider and expose its public
        # methods.
        for name in dir(item):
            if name.startswith("_"):
                continue
            try:
                member = getattr(item, name)
            except Exception:
                continue
            if callable(member) and not inspect.isclass(member):
                tools.append(member)
    return Toolset(tools)
