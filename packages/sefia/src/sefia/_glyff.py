import functools
from collections.abc import Callable
from typing import Any, Awaitable, ParamSpec, TypeVar, cast

import glyff

P = ParamSpec("P")
R = TypeVar("R")

RUNTIME_DOMAIN = glyff.Domain("sefia.runtime", version="1")


def engrave(
    domain: glyff.Domain,
    func: Callable[P, Awaitable[R]],
    *,
    name: str | None = None,
) -> Callable[P, Awaitable[R]]:
    """Bind an engraved callable to its qualified or explicit persisted name."""

    @functools.wraps(func)
    async def named(*args: P.args, **kwargs: P.kwargs) -> Any:
        return await func(*args, **kwargs)

    if name is not None:
        named.__qualname__ = name
    return cast(Callable[P, Awaitable[R]], domain.engrave(named))
