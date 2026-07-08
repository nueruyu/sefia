from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

from ._events import SessionEventBroker, format_sse_event


def session_event_response(
    broker: SessionEventBroker, session_id: str
) -> StreamingResponse:
    """A ``text/event-stream`` response relaying one session's events."""
    return StreamingResponse(
        _event_stream(broker, session_id),
        media_type="text/event-stream",
    )


async def _event_stream(
    broker: SessionEventBroker, session_id: str
) -> AsyncIterator[str]:
    async with broker.subscribe(session_id) as queue:
        while True:
            event = await queue.get()
            yield format_sse_event(event.name, event.data)
