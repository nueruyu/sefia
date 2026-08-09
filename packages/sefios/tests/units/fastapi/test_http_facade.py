import asyncio

import pytest
from sefia_fastapi import UnknownSessionError as HTTPUnknownSessionError
from sefios.fastapi import SefiaHTTP
from sefios.tools import Input, Output, OutputMessage


@pytest.fixture
def http(tmp_path) -> SefiaHTTP:
    return SefiaHTTP(session_dir=tmp_path / "sessions", model="gpt-4o-mini")


def _drain(queue: asyncio.Queue) -> list:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


class TestSefiaHTTPSessionManagement:
    def test_input_tool_is_exposed(self, http: SefiaHTTP):
        assert isinstance(http.input_tool, Input)

    def test_output_tool_is_exposed(self, http: SefiaHTTP):
        assert isinstance(http.output_tool, Output)

    def test_created_session_is_known(self, http: SefiaHTTP):
        session_id = http.create_session()

        http.ensure_session(session_id)

    def test_unknown_session_raises_http_error(self, http: SefiaHTTP):
        # The facade raises the sefia_fastapi exception that applications map
        # to an HTTP response.
        with pytest.raises(HTTPUnknownSessionError) as exc_info:
            http.ensure_session("ghost")

        assert exc_info.value.session_id == "ghost"

    def test_events_for_unknown_session_raises(self, http: SefiaHTTP):
        with pytest.raises(HTTPUnknownSessionError):
            http.events("ghost")

    def test_events_for_known_session_returns_sse_response(self, http: SefiaHTTP):
        session_id = http.create_session()

        response = http.events(session_id)

        assert response.media_type == "text/event-stream"

    async def test_session_for_unknown_session_raises(self, http: SefiaHTTP):
        with pytest.raises(HTTPUnknownSessionError):
            async with http.session(session_id="ghost"):
                pass


class TestSefiaHTTPOutput:
    async def test_send_output_publishes_output_event_to_its_session(
        self, http: SefiaHTTP
    ):
        session_id = http.create_session()

        async with http._events._subscribe(session_id) as queue:
            async with http.session(session_id=session_id):
                await http._emit_output(
                    OutputMessage(interaction_id="call-1", message="Hello there!")
                )

        outputs = [event for event in _drain(queue) if event.name == "output"]
        assert len(outputs) == 1
        assert outputs[0].data["message"] == "Hello there!"
        assert outputs[0].data["interaction_id"]

    async def test_concurrent_sessions_do_not_mix_output_events(self, http: SefiaHTTP):
        first = http.create_session()
        second = http.create_session()

        async def emit(session_id: str, message: str) -> None:
            async with http.session(session_id=session_id):
                await http._emit_output(
                    OutputMessage(interaction_id=session_id, message=message)
                )

        async with (
            http._events._subscribe(first) as first_queue,
            http._events._subscribe(second) as second_queue,
        ):
            await asyncio.gather(
                emit(first, "for the first"),
                emit(second, "for the second"),
            )

        first_outputs = [e for e in _drain(first_queue) if e.name == "output"]
        second_outputs = [e for e in _drain(second_queue) if e.name == "output"]
        assert [e.data["message"] for e in first_outputs] == ["for the first"]
        assert [e.data["message"] for e in second_outputs] == ["for the second"]

    async def test_emit_output_without_a_session_raises(self, http: SefiaHTTP):
        with pytest.raises(RuntimeError):
            await http._emit_output(
                OutputMessage(interaction_id="x", message="orphaned")
            )


class TestSefiaHTTPDeltas:
    async def test_output_deltas_use_the_completion_interaction_id(
        self, http: SefiaHTTP
    ):
        session_id = http.create_session()

        async with http._events._subscribe(session_id) as queue:
            async with http.session(session_id=session_id):
                await http._emit_output_delta("call-1", "Hel")
                await http._emit_output_delta("call-1", "lo")
                await http._emit_output(
                    OutputMessage(interaction_id="call-1", message="Hello")
                )

        events = _drain(queue)
        deltas = [e for e in events if e.name == "delta"]
        assert all(e.data["type"] == "output" for e in deltas)
        assert [e.data["text"] for e in deltas] == ["Hel", "lo"]
        assert {e.data["interaction_id"] for e in deltas} == {"call-1"}
        outputs = [e for e in events if e.name == "output"]
        assert outputs[0].data["interaction_id"] == "call-1"

    async def test_input_deltas_use_their_tool_call_ids(self, http: SefiaHTTP):
        session_id = http.create_session()

        async with http._events._subscribe(session_id) as queue:
            async with http.session(session_id=session_id):
                await http._emit_input_delta("call-1", "Q1")
                await http._emit_input_delta("call-2", "Q2")

        deltas = [e for e in _drain(queue) if e.name == "delta"]
        assert all(e.data["type"] == "input" for e in deltas)
        assert [e.data["text"] for e in deltas] == ["Q1", "Q2"]
        assert [e.data["interaction_id"] for e in deltas] == ["call-1", "call-2"]

    async def test_input_and_output_deltas_keep_independent_bubble_ids(
        self, http: SefiaHTTP
    ):
        session_id = http.create_session()

        async with http._events._subscribe(session_id) as queue:
            async with http.session(session_id=session_id):
                await http._emit_output_delta("output-call", "narrating")
                await http._emit_input_delta("input-call", "asking")

        deltas = {e.data["type"]: e.data for e in _drain(queue) if e.name == "delta"}
        assert deltas["output"]["interaction_id"] != deltas["input"]["interaction_id"]
