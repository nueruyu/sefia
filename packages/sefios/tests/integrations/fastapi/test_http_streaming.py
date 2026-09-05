import asyncio
import json
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.responses import StreamingResponse
from typing_extensions import override

from sefia import Tools
from sefia.llm import LLMClient, LLMCompletion, Message
from sefia.llm.step_decision import DecisionSpec, StepTool
from sefia.llm.streaming import OutputStreamCallback, StringDelta, StringEnd
from sefia.llm.structured_data import StructuredData
from sefia_fastapi.events import SSEEvent
from sefios import domain
from sefios.exceptions import InputRequired
from sefios.fastapi import SefiaHTTP
from sefios.tools import Input, Output

infer = domain(
    "packages.sefios.tests.integrations.fastapi.test_http_streaming", version="1"
).infer


class _StreamingClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    @override
    async def complete(
        self,
        messages: list[Message],
        tools: list[StepTool] | None = None,
        decision_spec: DecisionSpec | None = None,
        stream_callback: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        output_callback: OutputStreamCallback | None = None,
        reasoning_callback: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> LLMCompletion:
        content = self.responses.pop(0)
        if output_callback is not None:
            payload = json.loads(content)
            for index, call in enumerate(payload.get("tool_calls", [])):
                await output_callback(
                    StringEnd(("tool_calls", index, "name"), call["name"])
                )
                for name, value in call["arguments"].items():
                    if not isinstance(value, str):
                        continue
                    path = ("tool_calls", index, "arguments", name)
                    for character in value:
                        await output_callback(StringDelta(path, character))
                    await output_callback(StringEnd(path, value))
        return LLMCompletion(
            content=content,
            structured_output=(
                StructuredData.parse_json(content)
                if decision_spec is not None
                else None
            ),
        )


class _OutputAgent:
    output: Tools[Output]

    def __init__(self, output: Output) -> None:
        self.output = output

    @infer
    async def run(self) -> str: ...


class _InputAgent:
    input: Tools[Input]

    def __init__(self, input_tool: Input) -> None:
        self.input = input_tool

    @infer
    async def run(self) -> str: ...


class _InputOutputAgent:
    input: Tools[Input]
    output: Tools[Output]

    def __init__(self, input_tool: Input, output: Output) -> None:
        self.input = input_tool
        self.output = output

    @infer
    async def run(self) -> str: ...


def _tool_response(*calls: tuple[str, dict[str, Any]]) -> str:
    return json.dumps(
        {
            "decision": "tool_calls",
            "tool_calls": [
                {"name": name, "arguments": arguments} for name, arguments in calls
            ],
        }
    )


@dataclass(frozen=True)
class _Event:
    name: str
    data: dict[str, Any]


async def _read_until(response: StreamingResponse, terminal_event: str) -> list[_Event]:
    events: list[_Event] = []
    async for chunk in response.body_iterator:
        assert isinstance(chunk, str)
        lines = chunk.strip().splitlines()
        event = _Event(
            name=lines[0].removeprefix("event: "),
            data=json.loads(lines[1].removeprefix("data: ")),
        )
        events.append(event)
        if event.name == terminal_event:
            return events
    raise AssertionError(f"SSE stream ended before {terminal_event!r}")


async def _start_reader(
    http: SefiaHTTP, session_id: str, terminal_event: str
) -> asyncio.Task[list[_Event]]:
    task = asyncio.create_task(_read_until(http.events(session_id), terminal_event))
    await asyncio.sleep(0)
    return task


async def test_output_tool_publishes_deltas_and_output_over_public_sse() -> None:
    http = SefiaHTTP(
        llm_client=_StreamingClient(
            [
                _tool_response(("Output_send_output", {"message": "Hello"})),
                json.dumps({"decision": "result", "result": "done"}),
            ]
        )
    )
    session_id = http.create_session()
    reader = await _start_reader(http, session_id, SSEEvent.COMPLETED)

    async with http.session(session_id=session_id):
        result = await _OutputAgent(http.output_tool).run()

    events = await reader
    deltas = [event.data for event in events if event.name == SSEEvent.DELTA]
    outputs = [event.data for event in events if event.name == SSEEvent.OUTPUT]
    assert result == "done"
    assert "".join(delta["text"] for delta in deltas) == "Hello"
    assert {delta["type"] for delta in deltas} == {"output"}
    assert len(outputs) == 1
    assert {delta["interaction_id"] for delta in deltas} == {
        outputs[0]["interaction_id"]
    }


async def test_input_tool_publishes_deltas_and_pause_over_public_sse() -> None:
    http = SefiaHTTP(
        llm_client=_StreamingClient(
            [_tool_response(("Input_get_input", {"prompt": "Your name?"}))]
        )
    )
    session_id = http.create_session()
    reader = await _start_reader(http, session_id, SSEEvent.INPUT_REQUIRED)

    with pytest.raises(InputRequired) as pause:
        async with http.session(session_id=session_id):
            await _InputAgent(http.input_tool).run()

    events = await reader
    deltas = [event.data for event in events if event.name == SSEEvent.DELTA]
    required = [event.data for event in events if event.name == SSEEvent.INPUT_REQUIRED]
    assert "".join(delta["text"] for delta in deltas) == "Your name?"
    assert {delta["type"] for delta in deltas} == {"input"}
    assert len(required) == 1
    assert {delta["interaction_id"] for delta in deltas} == {
        pause.value.interaction_id,
        required[0]["interaction_id"],
    }


async def test_input_and_output_deltas_use_independent_interaction_ids() -> None:
    http = SefiaHTTP(
        llm_client=_StreamingClient(
            [
                _tool_response(
                    ("Output_send_output", {"message": "Working"}),
                    ("Input_get_input", {"prompt": "Continue?"}),
                )
            ]
        )
    )
    session_id = http.create_session()
    reader = await _start_reader(http, session_id, SSEEvent.INPUT_REQUIRED)

    with pytest.raises(InputRequired):
        async with http.session(session_id=session_id):
            await _InputOutputAgent(http.input_tool, http.output_tool).run()

    events = await reader
    ids_by_type = {
        event.data["type"]: event.data["interaction_id"]
        for event in events
        if event.name == SSEEvent.DELTA
    }
    assert ids_by_type["input"] != ids_by_type["output"]
