import contextvars
import inspect
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import glyff
from glyff.exceptions import ContextNotSetError

from .event_system import EventPublisher
from .events import ToolExecutionBound


@dataclass(frozen=True)
class _ToolCallContext:
    id: str
    function: Callable[..., Any] | None
    publisher: EventPublisher | None
    parent_id: glyff.ExecutionId | None


_tool_call_var = contextvars.ContextVar[_ToolCallContext]("sefia_tool_call")


def _callable_identity(
    function: Callable[..., Any] | None,
) -> Callable[..., Any] | None:
    if function is None:
        return None
    return getattr(function, "__func__", function)


@contextmanager
def serving_tool_call(
    call_id: str,
    function: Callable[..., Any] | None = None,
    publisher: EventPublisher | None = None,
) -> Generator[None]:
    """Bind ``call_id`` as the tool call the current handler is serving."""
    try:
        parent_id = glyff.get_context().current_execution_id
    except ContextNotSetError:
        parent_id = None
    token = _tool_call_var.set(
        _ToolCallContext(
            id=call_id,
            function=_callable_identity(function),
            publisher=publisher,
            parent_id=parent_id,
        )
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


P = ParamSpec("P")
R = TypeVar("R")


def bind_tool_execution(
    function: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Observe a dispatched callable after glyff assigns its execution identity."""

    @wraps(function)
    async def bound(*args: P.args, **kwargs: P.kwargs) -> R:
        serving = _tool_call_var.get(None)
        execution_id = glyff.get_context().current_execution_id
        if (
            serving is not None
            and serving.publisher is not None
            and serving.function is not None
            and inspect.unwrap(serving.function) is inspect.unwrap(function)
            and execution_id is not None
            and execution_id.parent_id == serving.parent_id
        ):
            await serving.publisher.publish(
                ToolExecutionBound(tool_call_id=serving.id, execution_id=execution_id)
            )
        return await function(*args, **kwargs)

    return bound
