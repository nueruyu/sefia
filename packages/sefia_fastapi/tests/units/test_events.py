import asyncio
from dataclasses import dataclass

from sefia.llm.events import LLMTokenReceived
from sefia_fastapi import SessionEvents, SSEEvent


class TestSSEEvent:
    def test_names_are_the_wire_contract(self):
        assert SSEEvent.TOKEN == "token"
        assert SSEEvent.INPUT_REQUIRED == "input_required"
        assert SSEEvent.COMPLETED == "completed"
        assert SSEEvent.EXECUTION_FAILED == "execution_failed"


class TestPublish:
    async def test_publish_without_subscribers_is_a_noop(self):
        events = SessionEvents()

        await events.publish("s1", "token", "hello")

    async def test_subscriber_receives_published_events(self):
        events = SessionEvents()

        async with events._subscribe("s1") as queue:
            await events.publish("s1", "token", "hello")

            event = queue.get_nowait()

        assert event.name == "token"
        assert event.data == "hello"

    async def test_events_are_scoped_to_their_session(self):
        events = SessionEvents()

        async with events._subscribe("s1") as queue:
            await events.publish("other", "token", "hello")

            assert queue.empty()

    async def test_unsubscribed_queue_stops_receiving(self):
        events = SessionEvents()

        async with events._subscribe("s1") as queue:
            pass
        await events.publish("s1", "token", "late")

        assert queue.empty()


class TestTokenHandler:
    async def test_relays_tokens_to_the_session_stream(self):
        events = SessionEvents()
        handler = events.token_handler("s1")

        async with events._subscribe("s1") as queue:
            await handler.handle(LLMTokenReceived(token="hi"))

            event = queue.get_nowait()

        assert event.name == SSEEvent.TOKEN
        assert event.data == "hi"


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

        events = SessionEvents()
        response = events.response("s1")

        stream = response.body_iterator.__aiter__()
        first_chunk = asyncio.ensure_future(stream.__anext__())
        await asyncio.sleep(0)

        await events.publish("s1", "input_required", {"request": Payload("x")})

        chunk = await asyncio.wait_for(first_chunk, timeout=1)
        assert isinstance(chunk, str)
        assert '"interaction_id": "x"' in chunk
