"""Per-session server-sent events: publish, token relay, and the SSE response."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from fastapi.responses import StreamingResponse
from sefia.event_system import EventHandler
from sefia.llm.events import LLMTokenReceived


class SSEEvent:
    """The wire names of the server-sent events an application publishes.

    Single source of truth: the facade and browser clients import these rather
    than repeating literals.
    """

    TOKEN = "token"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    EXECUTION_FAILED = "execution_failed"


@dataclass(frozen=True)
class _SessionEvent:
    name: str
    data: Any


class SessionEvents:
    """Per-session event streams for an HTTP application.

    One object owns the whole surface: :meth:`publish` fans an event out to a
    session's subscribers, :meth:`token_handler` returns a sefia event handler
    that relays LLM tokens into the stream, and :meth:`response` serves the
    stream as a ``text/event-stream`` response. Publishing to a session nobody
    is subscribed to is a no-op.
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

    def token_handler(self, session_id: str) -> EventHandler[LLMTokenReceived]:
        """A sefia event handler relaying LLM tokens into this session's stream."""
        return _TokenRelay(self, session_id)

    def response(self, session_id: str) -> StreamingResponse:
        """A ``text/event-stream`` response relaying this session's events."""
        return StreamingResponse(
            self._event_stream(session_id),
            media_type="text/event-stream",
        )

    @asynccontextmanager
    async def _subscribe(
        self, session_id: str
    ) -> AsyncIterator[asyncio.Queue[_SessionEvent]]:
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


class _TokenRelay(EventHandler[LLMTokenReceived]):
    def __init__(self, events: SessionEvents, session_id: str):
        self._events = events
        self._session_id = session_id

    async def handle(self, event: LLMTokenReceived) -> None:
        await self._events.publish(self._session_id, SSEEvent.TOKEN, event.token)


def _format_sse_event(event: str, data: Any) -> str:
    payload = json.dumps(_jsonable(data), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value
