import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

import pytest
from sefia import Tools
from sefia.llm import LLMClient, LLMResponse, Message
from sefia.llm.step_decision import StepDecisionModel
from sefia.llm.streaming import (
    OutputCallback,
    StringDelta,
    StringEnd,
)
from sefia_fastapi.events import _SessionEvent
from sefia_fastapi.exceptions import UnknownSessionError as HTTPUnknownSessionError
from sefios import MemoryPersistence, domain
from sefios.exceptions import InputRequired
from sefios.fastapi import SefiaHTTP
from sefios.tools import Input, Output, OutputMessage

infer = domain(
    "packages.sefios.tests.units.fastapi.test_http_facade", version="1"
).infer


class StreamingClient(LLMClient):
    def __init__(self, responses: list[str]):
        self.responses = responses

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        decision_model: StepDecisionModel | None = None,
        stream_callback: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        output_callback: OutputCallback | None = None,
        reasoning_callback: (Callable[[str], Coroutine[Any, Any, None]] | None) = None,
    ) -> LLMResponse:
        content = self.responses.pop(0)
        if stream_callback is not None:
            for character in content:
                await stream_callback(character)
        if output_callback is not None:
            payload = json.loads(content)
            for index, call in enumerate(payload.get("tool_calls", [])):
                await output_callback(
                    StringEnd(("tool_calls", index, "name"), call["name"])
                )
                for name, value in call["arguments"].items():
                    if isinstance(value, str):
                        for character in value:
                            await output_callback(
                                StringDelta(
                                    ("tool_calls", index, "arguments", name),
                                    character,
                                )
                            )
                        await output_callback(
                            StringEnd(
                                ("tool_calls", index, "arguments", name),
                                value,
                            )
                        )
        return LLMResponse(content=content)


class OutputAgent:
    _output: Tools[Output]

    def __init__(self, output: Output):
        self._output = output

    @infer
    async def run(self) -> str:
        """Send an output message."""
        ...


class InputAgent:
    _input: Tools[Input]

    def __init__(self, input_tool: Input):
        self._input = input_tool

    @infer
    async def run(self) -> str:
        """Ask for input."""
        ...


def _tool_response(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps(
        {
            "decision": "tool_calls",
            "tool_calls": [{"name": name, "arguments": arguments}],
        }
    )


@pytest.fixture
def http() -> SefiaHTTP:
    return SefiaHTTP(model="gpt-4o-mini")


def _drain(queue: asyncio.Queue[_SessionEvent]) -> list[_SessionEvent]:
    events: list[_SessionEvent] = []
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

    def test_created_session_is_known_to_another_instance(self) -> None:
        persistence = MemoryPersistence()
        first = SefiaHTTP(model="gpt-4o-mini", persistence=persistence)
        second = SefiaHTTP(model="gpt-4o-mini", persistence=persistence)

        session_id = first.create_session()

        second.ensure_session(session_id)

    def test_default_persistence_is_process_local(self) -> None:
        first = SefiaHTTP(model="gpt-4o-mini")
        second = SefiaHTTP(model="gpt-4o-mini")

        session_id = first.create_session()

        with pytest.raises(HTTPUnknownSessionError):
            second.ensure_session(session_id)

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
    async def test_streamed_output_deltas_match_the_completion_id(
        self, http: SefiaHTTP
    ):
        session_id = http.create_session()
        http._session_scope.llm_client = StreamingClient(
            [
                _tool_response("Output_send_output", {"message": "Hello"}),
                json.dumps({"decision": "result", "result": "done"}),
            ]
        )

        async with http._events._subscribe(session_id) as queue:
            async with http.session(session_id=session_id):
                await OutputAgent(http.output_tool).run()

        events = _drain(queue)
        deltas = [e for e in events if e.name == "delta"]
        assert all(e.data["type"] == "output" for e in deltas)
        assert "".join(e.data["text"] for e in deltas) == "Hello"
        outputs = [e for e in events if e.name == "output"]
        assert len(outputs) == 1
        assert {e.data["interaction_id"] for e in deltas} == {
            outputs[0].data["interaction_id"]
        }

    async def test_streamed_input_deltas_match_the_pause_id(self, http: SefiaHTTP):
        session_id = http.create_session()
        http._session_scope.llm_client = StreamingClient(
            [_tool_response("Input_get_input", {"prompt": "Your name?"})]
        )

        async with http._events._subscribe(session_id) as queue:
            with pytest.raises(InputRequired) as pause:
                async with http.session(session_id=session_id):
                    await InputAgent(http.input_tool).run()

        events = _drain(queue)
        deltas = [e for e in events if e.name == "delta"]
        assert all(e.data["type"] == "input" for e in deltas)
        assert "".join(e.data["text"] for e in deltas) == "Your name?"
        required = [e for e in events if e.name == "input_required"]
        assert len(required) == 1
        assert {e.data["interaction_id"] for e in deltas} == {
            pause.value.interaction_id,
            required[0].data["interaction_id"],
        }

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
