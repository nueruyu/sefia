import asyncio
import json
from collections.abc import Callable, Coroutine
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest
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
from typing_extensions import override

infer = domain(
    "packages.sefios.tests.integrations.fastapi.test_http_streaming", version="1"
).infer


async def _stream_completion(
    content: str,
    decision_spec: DecisionSpec | None,
    output_callback: OutputStreamCallback | None,
) -> LLMCompletion:
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
            StructuredData.parse_json(content) if decision_spec is not None else None
        ),
    )


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
        return await _stream_completion(content, decision_spec, output_callback)


class _ConcurrentStreamingClient(LLMClient):
    _messages = {
        "first-session-marker": "for the first",
        "second-session-marker": "for the second",
    }

    def __init__(self) -> None:
        self._calls = dict.fromkeys(self._messages, 0)
        self._waiting = 0
        self._both_started = asyncio.Event()

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
        prompt = "\n".join(
            message.content for message in messages if isinstance(message.content, str)
        )
        marker = next(marker for marker in self._messages if marker in prompt)
        self._calls[marker] += 1

        if self._calls[marker] == 1:
            self._waiting += 1
            if self._waiting == len(self._messages):
                self._both_started.set()
            await asyncio.wait_for(self._both_started.wait(), timeout=5)
            content = _tool_response(
                ("Output_send_output", {"message": self._messages[marker]})
            )
        else:
            content = json.dumps({"decision": "result", "result": "done"})

        return await _stream_completion(content, decision_spec, output_callback)


class _OutputAgent:
    output: Tools[Output]

    def __init__(self, output: Output) -> None:
        self.output = output

    @infer
    async def run(self) -> str: ...


class _ConcurrentOutputAgent:
    output: Tools[Output]

    def __init__(self, output: Output) -> None:
        self.output = output

    @infer
    async def run(self, marker: str) -> str: ...


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


async def test_output_tool_publishes_deltas_and_output_over_public_sse(
    read_sse: Callable[
        ..., AbstractAsyncContextManager[asyncio.Task[list[dict[str, Any]]]]
    ],
) -> None:
    http = SefiaHTTP(
        llm_client=_StreamingClient(
            [
                _tool_response(("Output_send_output", {"message": "Hello"})),
                json.dumps({"decision": "result", "result": "done"}),
            ]
        )
    )
    session_id = http.create_session()
    async with read_sse(http.events(session_id), SSEEvent.COMPLETED) as reader:
        async with http.session(session_id=session_id):
            result = await _OutputAgent(http.output_tool).run()

        events = await reader
        deltas = [event["data"] for event in events if event["name"] == SSEEvent.DELTA]
        outputs = [
            event["data"] for event in events if event["name"] == SSEEvent.OUTPUT
        ]
        assert result == "done"
        assert "".join(delta["text"] for delta in deltas) == "Hello"
        assert {delta["type"] for delta in deltas} == {"output"}
        assert len(outputs) == 1
        assert {delta["interaction_id"] for delta in deltas} == {
            outputs[0]["interaction_id"]
        }


async def test_concurrent_sessions_do_not_mix_public_output_events(
    read_sse: Callable[
        ..., AbstractAsyncContextManager[asyncio.Task[list[dict[str, Any]]]]
    ],
) -> None:
    http = SefiaHTTP(llm_client=_ConcurrentStreamingClient())
    first_session = http.create_session()
    second_session = http.create_session()
    async with (
        read_sse(http.events(first_session), SSEEvent.COMPLETED) as first_reader,
        read_sse(http.events(second_session), SSEEvent.COMPLETED) as second_reader,
    ):

        async def run(session_id: str, marker: str) -> str:
            async with http.session(session_id=session_id):
                return await _ConcurrentOutputAgent(http.output_tool).run(marker)

        results = await asyncio.gather(
            run(first_session, "first-session-marker"),
            run(second_session, "second-session-marker"),
        )
        first_events, second_events = await asyncio.gather(first_reader, second_reader)

        def output_texts(events: list[dict[str, Any]]) -> list[str]:
            return [
                event["data"]["message"]
                for event in events
                if event["name"] == SSEEvent.OUTPUT
            ]

        def delta_text(events: list[dict[str, Any]]) -> str:
            return "".join(
                event["data"]["text"]
                for event in events
                if event["name"] == SSEEvent.DELTA
            )

        assert results == ["done", "done"]
        assert output_texts(first_events) == ["for the first"]
        assert output_texts(second_events) == ["for the second"]
        assert delta_text(first_events) == "for the first"
        assert delta_text(second_events) == "for the second"


async def test_input_tool_publishes_deltas_and_pause_over_public_sse(
    read_sse: Callable[
        ..., AbstractAsyncContextManager[asyncio.Task[list[dict[str, Any]]]]
    ],
) -> None:
    http = SefiaHTTP(
        llm_client=_StreamingClient(
            [_tool_response(("Input_get_input", {"prompt": "Your name?"}))]
        )
    )
    session_id = http.create_session()
    async with read_sse(http.events(session_id), SSEEvent.INPUT_REQUIRED) as reader:
        with pytest.raises(InputRequired) as pause:
            async with http.session(session_id=session_id):
                await _InputAgent(http.input_tool).run()

        events = await reader
        deltas = [event["data"] for event in events if event["name"] == SSEEvent.DELTA]
        required = [
            event["data"]
            for event in events
            if event["name"] == SSEEvent.INPUT_REQUIRED
        ]
        assert "".join(delta["text"] for delta in deltas) == "Your name?"
        assert {delta["type"] for delta in deltas} == {"input"}
        assert len(required) == 1
        bindings = [
            event["data"] for event in events if event["name"] == SSEEvent.INPUT_BOUND
        ]
        assert len(bindings) == 1
        assert {delta["preview_id"] for delta in deltas} == {bindings[0]["preview_id"]}
        assert (
            bindings[0]["interaction_id"]
            == pause.value.interaction_id
            == required[0]["interaction_id"]
        )
        assert events.index(
            next(e for e in events if e["name"] == SSEEvent.INPUT_BOUND)
        ) < events.index(
            next(e for e in events if e["name"] == SSEEvent.INPUT_REQUIRED)
        )


async def test_input_and_output_deltas_use_independent_interaction_ids(
    read_sse: Callable[
        ..., AbstractAsyncContextManager[asyncio.Task[list[dict[str, Any]]]]
    ],
) -> None:
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
    async with read_sse(http.events(session_id), SSEEvent.INPUT_REQUIRED) as reader:
        with pytest.raises(InputRequired):
            async with http.session(session_id=session_id):
                await _InputOutputAgent(http.input_tool, http.output_tool).run()

        events = await reader
        ids_by_type = {
            event["data"]["type"]: event["data"].get(
                "preview_id", event["data"].get("interaction_id")
            )
            for event in events
            if event["name"] == SSEEvent.DELTA
        }
        assert ids_by_type["input"] != ids_by_type["output"]


async def test_application_input_publishes_complete_prompt_without_preview(
    read_sse: Callable[
        ..., AbstractAsyncContextManager[asyncio.Task[list[dict[str, Any]]]]
    ],
) -> None:
    from sefia.testing import MockLLMClient
    from sefios import require_input

    http = SefiaHTTP(llm_client=MockLLMClient([]))
    sid = http.create_session()
    async with read_sse(http.events(sid), SSEEvent.INPUT_REQUIRED) as reader:
        with pytest.raises(InputRequired) as pause:
            async with http.session(session_id=sid):
                await require_input("Approve?")
        events = await reader
    assert events == [
        {
            "name": SSEEvent.INPUT_REQUIRED,
            "data": {
                "prompt": "Approve?",
                "interaction_id": pause.value.interaction_id,
            },
        }
    ]
