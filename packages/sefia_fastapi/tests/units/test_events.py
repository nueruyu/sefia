import asyncio
from dataclasses import dataclass

from sefia.llm.events import LLMTokenReceived
from sefia_fastapi import (
    SessionEventBroker,
    TokenEventPublisher,
    format_sse_event,
    session_event_response,
)


class TestSessionEventBroker:
    async def test_publish_without_subscribers_is_a_noop(self):
        broker = SessionEventBroker()

        await broker.publish("s1", "token", "hello")

    async def test_subscriber_receives_published_events(self):
        broker = SessionEventBroker()

        async with broker.subscribe("s1") as queue:
            await broker.publish("s1", "token", "hello")

            event = queue.get_nowait()

        assert event.name == "token"
        assert event.data == "hello"

    async def test_events_are_scoped_to_their_session(self):
        broker = SessionEventBroker()

        async with broker.subscribe("s1") as queue:
            await broker.publish("other", "token", "hello")

            assert queue.empty()

    async def test_unsubscribed_queue_stops_receiving(self):
        broker = SessionEventBroker()

        async with broker.subscribe("s1") as queue:
            pass
        await broker.publish("s1", "token", "late")

        assert queue.empty()


class TestTokenEventPublisher:
    async def test_relays_tokens_to_broker(self):
        broker = SessionEventBroker()
        publisher = TokenEventPublisher(broker, "s1")

        async with broker.subscribe("s1") as queue:
            await publisher.handle(LLMTokenReceived(token="hi"))

            event = queue.get_nowait()

        assert event.name == "token"
        assert event.data == "hi"


class TestFormatSseEvent:
    def test_formats_event_and_json_data(self):
        assert format_sse_event("token", "hi") == 'event: token\ndata: "hi"\n\n'

    def test_serializes_dataclasses(self):
        @dataclass(frozen=True)
        class Payload:
            interaction_id: str

        formatted = format_sse_event("input_required", {"request": Payload("x")})

        assert '"interaction_id": "x"' in formatted


class TestSessionEventResponse:
    async def test_streams_published_events_as_sse(self):
        broker = SessionEventBroker()
        response = session_event_response(broker, "s1")
        assert response.media_type == "text/event-stream"

        stream = response.body_iterator.__aiter__()
        first_chunk = asyncio.ensure_future(stream.__anext__())
        # Yield control so the stream subscribes before the publish below.
        await asyncio.sleep(0)

        await broker.publish("s1", "completed", {"session_id": "s1"})

        chunk = await asyncio.wait_for(first_chunk, timeout=1)
        assert chunk == 'event: completed\ndata: {"session_id": "s1"}\n\n'
