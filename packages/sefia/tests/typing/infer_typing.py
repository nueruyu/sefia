from collections.abc import Awaitable
from typing import assert_type

from sefios import domain

infer = domain("typing").infer


@infer
async def async_inference(question: str) -> str:
    return question


result: Awaitable[str] = async_inference("question")
assert_type(result, Awaitable[str])


def sync_inference(question: str) -> str:
    return question


infer(sync_inference)  # pyright: ignore[reportArgumentType]
