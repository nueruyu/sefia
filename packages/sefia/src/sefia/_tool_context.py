import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

# Bound only for the duration of a single ``ToolEntry.invoke``. Kept private to
# this module; the setter and the accessor below are the only way across the
# boundary, so the raw contextvar never leaks.
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
    """The id of the tool call the current handler is serving.

    Read inside a tool body, this returns the id of the call being executed —
    the same ``ToolCallRequest.id`` across a pause and resume of that call. A
    transport-backed or client-side tool (an MCP round-trip, an HTTP-delegated
    or human-in-the-loop tool that pauses and resumes) uses it as a stable,
    replay-safe key to correlate a paused call with a later-provided result,
    without reaching into the durable-execution engine underneath.

    The id is bound only for the duration of the call's ``invoke``, in the
    current context. Like any ``contextvars`` value it is inherited by a task
    spawned during the call, so a background task started from a handler keeps
    reading the call's id after the handler returns; read it synchronously in
    the tool body when you need it scoped strictly to the call.

    Raises ``RuntimeError`` when no call is bound in the current context —
    outside a tool body, or from a task created before the call began.
    """
    try:
        return _tool_call_id_var.get()
    except LookupError:
        raise RuntimeError(
            "current_tool_call_id() is only available inside a tool call."
        ) from None
