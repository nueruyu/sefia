import inspect
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]


async def maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value
