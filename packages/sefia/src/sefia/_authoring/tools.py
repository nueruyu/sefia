from typing import Any, Awaitable, Callable, TypeVar

from .._tool_system import set_concurrent, set_stream_handler

C = TypeVar("C", bound=Callable[..., object])
StreamHandler = TypeVar("StreamHandler", bound=Callable[..., Awaitable[None]])


def preview(target: Callable[..., Any]) -> Callable[[StreamHandler], StreamHandler]:
    """Register a handler that previews a tool method's streamed arguments."""
    underlying = getattr(target, "__func__", target)

    def decorator(handler: StreamHandler) -> StreamHandler:
        set_stream_handler(underlying, handler)
        return handler

    return decorator


def concurrent(target: C) -> C:
    """Mark a tool method as safe to overlap with concurrent tool calls."""
    underlying = getattr(target, "__func__", target)
    set_concurrent(underlying)
    return target
