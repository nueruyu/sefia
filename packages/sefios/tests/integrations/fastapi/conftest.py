import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import aclosing, asynccontextmanager, suppress
from typing import Any, cast

import pytest
from fastapi.responses import StreamingResponse


async def _read_until(
    response: StreamingResponse, terminal_event: str
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    stream = response.body_iterator
    assert isinstance(stream, AsyncGenerator)
    async with aclosing(cast(AsyncGenerator[object, None], stream)):
        async for chunk in stream:
            assert isinstance(chunk, str)
            lines = chunk.strip().splitlines()
            event = {
                "name": lines[0].removeprefix("event: "),
                "data": json.loads(lines[1].removeprefix("data: ")),
            }
            events.append(event)
            if event["name"] == terminal_event:
                return events
    raise AssertionError(f"SSE stream ended before {terminal_event!r}")


@asynccontextmanager
async def _read_events(
    response: StreamingResponse, terminal_event: str, *, timeout: float = 5
) -> AsyncGenerator[asyncio.Task[list[dict[str, Any]]]]:
    async with asyncio.timeout(timeout):
        task = asyncio.create_task(_read_until(response, terminal_event))
        try:
            await asyncio.sleep(0)
            yield task
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.fixture
def read_sse():
    return _read_events
