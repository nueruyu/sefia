import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest
from fastapi.responses import StreamingResponse


@pytest.mark.parametrize("failure", ["timeout", "body"])
async def test_reader_cancels_task_and_closes_stream_on_failure(
    read_sse: Callable[
        ..., AbstractAsyncContextManager[asyncio.Task[list[dict[str, Any]]]]
    ],
    failure: str,
) -> None:
    closed = asyncio.Event()

    async def stream() -> AsyncGenerator[str]:
        try:
            yield "event: delta\ndata: {}\n\n"
            await asyncio.Future()
        finally:
            closed.set()

    response = StreamingResponse(stream())
    reader: asyncio.Task[list[dict[str, Any]]] | None = None
    with pytest.raises(TimeoutError if failure == "timeout" else RuntimeError):
        async with read_sse(response, "completed", timeout=0.05) as reader:
            if failure == "body":
                raise RuntimeError("test failed")
            await reader

    assert reader is not None and reader.cancelled()
    assert closed.is_set()
