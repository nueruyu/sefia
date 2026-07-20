import asyncio
from dataclasses import dataclass
from datetime import datetime

from sefia_fastapi import SessionEvents, SSEEvent


class TestSSEEvent:
    def test_names_are_the_wire_contract(self):
        assert SSEEvent.DELTA == "delta"
        assert SSEEvent.INPUT_REQUIRED == "input_required"
        assert SSEEvent.OUTPUT == "output"
        assert SSEEvent.COMPLETED == "completed"
        assert SSEEvent.EXECUTION_FAILED == "execution_failed"


class TestPublish:
    async def test_publish_without_subscribers_is_a_noop(self):
        events = SessionEvents()

        await events.publish("s1", "delta", "hello")

    async def test_subscriber_receives_published_events(self):
        events = SessionEvents()

        async with events._subscribe("s1") as queue:
            await events.publish("s1", "delta", "hello")

            event = queue.get_nowait()

        assert event.name == "delta"
        assert event.data == "hello"

    async def test_events_are_scoped_to_their_session(self):
        events = SessionEvents()

        async with events._subscribe("s1") as queue:
            await events.publish("other", "delta", "hello")

            assert queue.empty()

    async def test_unsubscribed_queue_stops_receiving(self):
        events = SessionEvents()

        async with events._subscribe("s1") as queue:
            pass
        await events.publish("s1", "delta", "late")

        assert queue.empty()


class TestResponse:
    async def test_streams_published_events_as_sse(self):
        events = SessionEvents()
        response = events.response("s1")
        assert response.media_type == "text/event-stream"

        stream = response.body_iterator.__aiter__()
        first_chunk = asyncio.ensure_future(stream.__anext__())
        # Yield control so the stream subscribes before the publish below.
        await asyncio.sleep(0)

        await events.publish("s1", "completed", {"session_id": "s1"})

        chunk = await asyncio.wait_for(first_chunk, timeout=1)
        assert chunk == 'event: completed\ndata: {"session_id": "s1"}\n\n'

    async def test_serializes_dataclass_payloads(self):
        @dataclass(frozen=True)
        class Payload:
            interaction_id: str
            created_at: datetime

        events = SessionEvents()
        response = events.response("s1")

        stream = response.body_iterator.__aiter__()
        first_chunk = asyncio.ensure_future(stream.__anext__())
        await asyncio.sleep(0)

        payload = Payload("x", datetime(2026, 7, 12, 6, 35, 48))
        await events.publish("s1", "input_required", {"request": payload})

        chunk = await asyncio.wait_for(first_chunk, timeout=1)
        assert isinstance(chunk, str)  # narrow str | bytes for the `in` checks
        assert '"interaction_id": "x"' in chunk
        assert '"created_at": "2026-07-12T06:35:48"' in chunk
