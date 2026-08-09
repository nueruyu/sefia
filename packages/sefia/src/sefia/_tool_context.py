import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _ToolCallContext:
    id: str
    function: Callable[..., Any] | None


_tool_call_var = contextvars.ContextVar[_ToolCallContext]("sefia_tool_call")


def _callable_identity(
    function: Callable[..., Any] | None,
) -> Callable[..., Any] | None:
    if function is None:
        return None
    return getattr(function, "__func__", function)


@contextmanager
def serving_tool_call(
    call_id: str, function: Callable[..., Any] | None = None
) -> Iterator[None]:
    """Bind ``call_id`` as the tool call the current handler is serving."""
    token = _tool_call_var.set(
        _ToolCallContext(id=call_id, function=_callable_identity(function))
    )
    try:
        yield
    finally:
        _tool_call_var.reset(token)


def current_tool_call_id() -> str:
    """The ``ToolCallRequest.id`` of the call the current handler is serving.

    Stable across the call's pause and resume, so a transport-backed or
    client-side tool can key a paused call to a later result without reaching
    into glyff. Bound only around ``invoke``; a task spawned during the call
    inherits it per normal ``contextvars`` semantics. Raises ``RuntimeError``
    when no call is bound in the current context.
    """
    try:
        return _tool_call_var.get().id
    except LookupError:
        raise RuntimeError(
            "current_tool_call_id() is only available inside a tool call."
        ) from None


def current_tool_call_id_for(function: Callable[..., Any]) -> str | None:
    """Return the call id only when ``function`` is the dispatched tool."""
    try:
        context = _tool_call_var.get()
    except LookupError:
        return None

    if context.function is not _callable_identity(function):
        return None
    return context.id
