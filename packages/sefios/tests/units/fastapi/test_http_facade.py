import asyncio

import pytest
from sefia_fastapi import UnknownSessionError as HTTPUnknownSessionError
from sefios.fastapi import SefiaHTTP
from sefios.tools import InputTool, OutputMessage, OutputTool


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
        assert isinstance(http.input_tool, InputTool)

    def test_output_tool_is_exposed(self, http: SefiaHTTP):
        assert isinstance(http.output_tool, OutputTool)

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
                await http.output_tool.send_output("Hello there!")

        outputs = [event for event in _drain(queue) if event.name == "output"]
        assert len(outputs) == 1
        assert outputs[0].data["message"] == "Hello there!"
        assert outputs[0].data["interaction_id"]

    async def test_concurrent_sessions_do_not_mix_output_events(self, http: SefiaHTTP):
        first = http.create_session()
        second = http.create_session()

        async def emit(session_id: str, message: str) -> None:
            async with http.session(session_id=session_id):
                await http.output_tool.send_output(message)

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
