import functools
from collections.abc import Callable
from typing import Any, Awaitable, ParamSpec, TypeVar, cast

import glyff

P = ParamSpec("P")
R = TypeVar("R")

RUNTIME_DOMAIN = glyff.Domain("sefios.runtime", version="1")


def engrave(name: str):
    def decorator(
        func: Callable[P, Awaitable[R]],
    ) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def named(*args: P.args, **kwargs: P.kwargs) -> Any:
            return await func(*args, **kwargs)

        named.__qualname__ = name
        engraved = RUNTIME_DOMAIN.engrave(named)
        functools.update_wrapper(engraved, func)
        return cast(Callable[P, Awaitable[R]], engraved)

    return decorator
