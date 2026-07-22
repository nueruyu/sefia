import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

# Reached only through the accessors below, so the raw contextvar never leaks.
_tool_call_id_var = contextvars.ContextVar[str]("sefia_tool_call_id")


@contextmanager
def serving_tool_call(call_id: str) -> Iterator[None]:
    """Bind ``call_id`` as the tool call the current handler is serving."""
    token = _tool_call_id_var.set(call_id)
    try:
        yield
    finally:
        _tool_call_id_var.reset(token)


def current_tool_call_id() -> str:
    """The ``ToolCallRequest.id`` of the call the current handler is serving.

    Stable across the call's pause and resume, so a transport-backed or
    client-side tool can key a paused call to a later result without reaching
    into glyff. Bound only around ``invoke``; a task spawned during the call
    inherits it per normal ``contextvars`` semantics. Raises ``RuntimeError``
    when no call is bound in the current context.
    """
    try:
        return _tool_call_id_var.get()
    except LookupError:
        raise RuntimeError(
            "current_tool_call_id() is only available inside a tool call."
        ) from None
