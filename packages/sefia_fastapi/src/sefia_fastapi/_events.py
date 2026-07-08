"""Session-scoped server-sent events: broker, token relay, and SSE formatting."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from sefia.event_system import EventHandler
from sefia.llm.events import LLMTokenReceived


@dataclass(frozen=True)
class SessionEvent:
    """A named event published for one session."""

    name: str
    data: Any


class SessionEventBroker:
    """Fan-out of per-session events to any number of SSE subscribers."""

    def __init__(self):
        self._subscribers: dict[str, set[asyncio.Queue[SessionEvent]]] = {}

    @asynccontextmanager
    async def subscribe(
        self, session_id: str
    ) -> AsyncIterator[asyncio.Queue[SessionEvent]]:
        queue: asyncio.Queue[SessionEvent] = asyncio.Queue()
        subscribers = self._subscribers.setdefault(session_id, set())
        subscribers.add(queue)
        try:
            yield queue
        finally:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(session_id, None)

    async def publish(self, session_id: str, name: str, data: Any) -> None:
        subscribers = list(self._subscribers.get(session_id, ()))
        if not subscribers:
            return
        event = SessionEvent(name=name, data=data)
        for queue in subscribers:
            await queue.put(event)


class TokenEventPublisher(EventHandler[LLMTokenReceived]):
    """Relays LLM tokens for one session into a :class:`SessionEventBroker`."""

    def __init__(self, broker: SessionEventBroker, session_id: str):
        self._broker = broker
        self._session_id = session_id

    async def handle(self, event: LLMTokenReceived) -> None:
        await self._broker.publish(self._session_id, "token", event.token)


def format_sse_event(event: str, data: Any) -> str:
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
