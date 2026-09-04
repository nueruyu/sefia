import json
from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import Mock

import pytest

from sefia._tool_system import ToolRegistry
from sefia.event_system import EventPublisher
from sefia.inference import FunctionInfo, ToolCallsDecision
from sefia.llm import LLMInferenceStrategy, LLMResponse, Message
from sefia.llm.llm_output import LLMOutput
from sefia.llm._arg_stream import ToolArgStreamer, _ArgStreamChannel
from sefia.llm._client import LLMClient
from sefia.llm.step_decision import DecisionSpec, StepTool
from sefia.llm.streaming import (
    OutputStreamCallback,
    Scalar as OutputScalar,
    StringDelta as OutputStringDelta,
    StringEnd as OutputStringEnd,
)
from sefia.llm.transports import (
    DecisionTransport,
    PromptedDecisionTransport,
    StructuredDecisionTransport,
)
from sefia.pydantic import PydanticModelBackend
from sefia.streaming import (
    ArgStream,
    Scalar,
    StringDelta,
    StringEnd,
)


class Collector:
    """A stream handler that records every event it receives."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.tool_call_ids: list[str] = []

    async def __call__(self, tool_call_id: str, stream: ArgStream) -> None:
        self.tool_call_ids.append(tool_call_id)
        async for event in stream:
            self.events.append(event)


def _tool_call_id(index: int) -> str:
    return f"call-{index}"


async def run_router(
    json_text: str, *, chunk_size: int = 10_000, tool: str = "ask_human"
) -> list[Any]:
    collector = Collector()
    streamer = ToolArgStreamer({tool: collector}, _tool_call_id)

    _feed_events(streamer, json_text, chunk_size)
    await streamer.close()
    return collector.events


async def run_router_handlers(
    json_text: str, tools: list[str], *, chunk_size: int = 10_000
) -> dict[str, list[Any]]:
    collectors = {name: Collector() for name in tools}
    streamer = ToolArgStreamer(dict(collectors), _tool_call_id)

    _feed_events(streamer, json_text, chunk_size)
    await streamer.close()
    return {name: collector.events for name, collector in collectors.items()}


def _delta_text(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, StringDelta))


def _feed_events(streamer: ToolArgStreamer, text: str, chunk_size: int) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return
    for index, call in enumerate(payload.get("tool_calls", [])):
        for field, field_value in call.items():
            if field == "name":
                streamer.identify_tool(index, field_value)
            elif field == "arguments":
                for name, value in field_value.items():
                    if isinstance(value, str):
                        for start in range(0, len(value), chunk_size):
                            streamer.on_argument(
                                index,
                                StringDelta(
                                    name=name, text=value[start : start + chunk_size]
                                ),
                            )
                        streamer.on_argument(index, StringEnd(name=name, value=value))
                    else:
                        streamer.on_argument(index, Scalar(name=name, value=value))


# --- router unit tests --------------------------------------------------------


async def test_streams_string_argument():
    text = (
        '{"decision":"tool_calls","tool_calls":[{"name":"ask_human",'
        '"arguments":{"question":"Hello world"}}]}'
    )
    events = await run_router(text)

    assert _delta_text(events) == "Hello world"
    assert all(
        isinstance(e, StringDelta) and e.name == "question"
        for e in events
        if isinstance(e, StringDelta)
    )
    assert events[-1] == StringEnd(name="question", value="Hello world")


async def test_streams_string_argument_character_by_character():
    text = (
        '{"decision":"tool_calls","tool_calls":[{"name":"ask_human",'
        '"arguments":{"question":"Hello world"}}]}'
    )
    events = await run_router(text, chunk_size=1)

    assert _delta_text(events) == "Hello world"
    assert events[-1] == StringEnd(name="question", value="Hello world")


async def test_streams_scalar_arguments_whole():
    text = (
        '{"decision":"tool_calls","tool_calls":[{"name":"ask_human",'
        '"arguments":{"count":42,"ok":true,"note":null}}]}'
    )
    events = await run_router(text)

    assert Scalar(name="count", value=42) in events
    assert Scalar(name="ok", value=True) in events
    assert Scalar(name="note", value=None) in events
    assert not any(isinstance(e, StringDelta) for e in events)


async def test_resolves_when_arguments_precede_name():
    # JSON member order is the model's choice; routing must not depend on the
    # name arriving before the arguments.
    text = (
        '{"decision":"tool_calls","tool_calls":[{"arguments":{"question":"Hi there"},'
        '"name":"ask_human"}]}'
    )
    events = await run_router(text)

    assert _delta_text(events) == "Hi there"
    assert StringEnd(name="question", value="Hi there") in events


async def test_unregistered_tool_is_ignored():
    text = (
        '{"decision":"tool_calls","tool_calls":[{"name":"other",'
        '"arguments":{"question":"Hi"}}]}'
    )
    events = await run_router(text)  # router only knows ask_human

    assert events == []


async def test_duplicate_tool_name_resolution_is_ignored():
    async def handler(tool_call_id: str, stream: ArgStream) -> None:
        async for _ in stream:
            pass

    streamer = ToolArgStreamer({"ask_human": handler}, _tool_call_id)
    streamer.identify_tool(0, "ask_human")
    first_channel = streamer._channels[0]
    first_tasks = list(streamer._tasks)

    streamer.identify_tool(0, "ask_human")

    assert streamer._channels[0] is first_channel
    assert streamer._tasks == first_tasks
    await streamer.close()


async def test_result_is_not_streamed():
    text = '{"decision":"result","result":"the answer"}'
    events = await run_router(text)

    assert events == []


async def test_fenced_response_is_ignored_gracefully():
    # A markdown-fenced response is not the bare JSON the side channel parses;
    # it must stop quietly rather than raise.
    text = (
        "```json\n"
        '{"decision":"tool_calls","tool_calls":[{"name":"ask_human","arguments":{"question":"Hi"}}]}\n'
        "```"
    )
    events = await run_router(text)

    assert events == []


async def test_routes_multiple_tool_calls_independently():
    text = (
        '{"decision":"tool_calls","tool_calls":['
        '{"name":"ask_a","arguments":{"question":"A?"}},'
        '{"name":"ask_b","arguments":{"question":"B?"}}'
        "]}"
    )
    events = await run_router_handlers(text, ["ask_a", "ask_b"], chunk_size=3)

    assert StringEnd(name="question", value="A?") in events["ask_a"]
    assert StringEnd(name="question", value="B?") in events["ask_b"]
    # No cross-talk between the two calls' streams.
    assert _delta_text(events["ask_a"]) == "A?"
    assert _delta_text(events["ask_b"]) == "B?"


async def test_routes_multiple_arguments_of_one_call():
    text = (
        '{"decision":"tool_calls","tool_calls":[{"name":"ask",'
        '"arguments":{"question":"Hi","count":3}}]}'
    )
    events = (await run_router_handlers(text, ["ask"], chunk_size=4))["ask"]

    # A single multiplexed stream carries both arguments, distinguished by name.
    assert _delta_text(events) == "Hi"
    assert StringEnd(name="question", value="Hi") in events
    assert Scalar(name="count", value=3) in events


# --- strategy integration -----------------------------------------------------


class StreamingClient(LLMClient):
    """A fake client that streams its fixed content one character at a time."""

    def __init__(self, content: str) -> None:
        self.content = content

    async def complete(
        self,
        messages: list[Message],
        tools: list[StepTool] | None = None,
        decision_model: DecisionSpec | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
        output_callback: OutputStreamCallback | None = None,
        reasoning_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
    ) -> LLMResponse:
        if stream_callback is not None:
            for char in self.content:
                await stream_callback(char)
        if output_callback is not None:
            payload = json.loads(self.content)
            for index, call in enumerate(payload.get("tool_calls", [])):
                await output_callback(
                    OutputStringEnd(("tool_calls", index, "name"), call["name"])
                )
                for name, value in call["arguments"].items():
                    path = ("tool_calls", index, "arguments", name)
                    if isinstance(value, str):
                        for character in value:
                            await output_callback(OutputStringDelta(path, character))
                        await output_callback(OutputStringEnd(path, value))
                    else:
                        await output_callback(OutputScalar(path, value))
        return LLMResponse(
            content=self.content,
            structured_output=(
                LLMOutput.parse_json(self.content)
                if decision_model is not None
                else None
            ),
        )


def _function_info() -> FunctionInfo:
    return FunctionInfo(
        qualname="step",
        instructions="do it",
        bound_arguments={},
        type_hints={},
        return_type=str,
        args=(),
        kwargs={},
    )


_TOOL_CALL_CONTENT = (
    '{"decision": "tool_calls", "tool_calls": [{"name": "ask_human", '
    '"arguments": {"question": "What is your name?"}}]}'
)


@pytest.mark.parametrize(
    ("transport", "content"),
    [
        (StructuredDecisionTransport(), _TOOL_CALL_CONTENT),
        (
            PromptedDecisionTransport(),
            f"Decision with {{irrelevant}} braces:\n```json\n{_TOOL_CALL_CONTENT}\n```",
        ),
    ],
)
async def test_arguments_stream_through_a_real_strategy(
    transport: DecisionTransport,
    content: str,
) -> None:
    renderer = Mock()
    renderer.render.return_value = "prompt"
    strategy = LLMInferenceStrategy(
        llm_client=StreamingClient(content),
        result_format_factory=PydanticModelBackend(),
        prompt_renderer=renderer,
        decision_transport=transport,
        stream=True,
    )

    async def ask_human(question: str) -> str:
        return question

    collector = Collector()
    publisher = EventPublisher([])
    tools = ToolRegistry()
    tools.add(ask_human, name="ask_human", stream_handler=collector)

    decision = await strategy.decide_next_step(
        function_info=_function_info(),
        history=[],
        tools=tools,
        publisher=publisher,
    )

    assert isinstance(decision, ToolCallsDecision)
    assert collector.tool_call_ids == [decision.calls[0].id]
    assert (
        "".join(e.text for e in collector.events if isinstance(e, StringDelta))
        == "What is your name?"
    )
    assert StringEnd(name="question", value="What is your name?") in collector.events


async def test_channel_delivers_in_order_then_stops():
    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="q", text="a"))
    channel.feed(StringDelta(name="q", text="b"))
    channel.close()

    seen = [event async for event in channel]

    assert seen == [
        StringDelta(name="q", text="a"),
        StringDelta(name="q", text="b"),
    ]
