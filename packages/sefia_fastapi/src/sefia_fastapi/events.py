"""Per-session server-sent events: publish and the SSE response."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse


class SSEEvent:
    """The wire names of the server-sent events an application publishes.

    Single source of truth: the facade and browser clients import these rather
    than repeating literals.
    """

    DELTA = "delta"
    INPUT_REQUIRED = "input_required"
    OUTPUT = "output"
    COMPLETED = "completed"
    EXECUTION_FAILED = "execution_failed"


@dataclass(frozen=True)
class _SessionEvent:
    name: str
    data: Any


class SessionEvents:
    """Per-session event streams for an HTTP application.

    One object owns the whole surface: :meth:`publish` fans an event out to a
    session's subscribers and :meth:`response` serves the stream as a
    ``text/event-stream`` response. Publishing to a session nobody is
    subscribed to is a no-op.
    """

    def __init__(self):
        self._subscribers: dict[str, set[asyncio.Queue[_SessionEvent]]] = {}

    async def publish(self, session_id: str, name: str, data: Any) -> None:
        subscribers = list(self._subscribers.get(session_id, ()))
        if not subscribers:
            return
        event = _SessionEvent(name=name, data=data)
        for queue in subscribers:
            await queue.put(event)

    def response(self, session_id: str) -> StreamingResponse:
        """A ``text/event-stream`` response relaying this session's events."""
        return StreamingResponse(
            self._event_stream(session_id),
            media_type="text/event-stream",
        )

    @asynccontextmanager
    async def _subscribe(
        self, session_id: str
    ) -> AsyncGenerator[asyncio.Queue[_SessionEvent]]:
        queue: asyncio.Queue[_SessionEvent] = asyncio.Queue()
        subscribers = self._subscribers.setdefault(session_id, set())
        subscribers.add(queue)
        try:
            yield queue
        finally:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(session_id, None)

    async def _event_stream(self, session_id: str) -> AsyncIterator[str]:
        async with self._subscribe(session_id) as queue:
            while True:
                event = await queue.get()
                yield _format_sse_event(event.name, event.data)


def _format_sse_event(event: str, data: Any) -> str:
    payload = json.dumps(jsonable_encoder(data), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
