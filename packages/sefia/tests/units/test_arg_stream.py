from typing import Any

from sefia.llm._arg_stream import ToolArgStreamer
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
    events: list[tuple[int, str | Scalar | StringDelta | StringEnd]],
    *,
    tool: str = "ask_human",
) -> Collector:
    collector = Collector()
    streamer = ToolArgStreamer({tool: collector}, _tool_call_id)

    for index, event in events:
        if isinstance(event, str):
            streamer.identify_tool(index, event)
        else:
            streamer.on_argument(index, event)
    await streamer.close()
    return collector


async def run_router_handlers(
    events: list[tuple[int, str | Scalar | StringDelta | StringEnd]], tools: list[str]
) -> dict[str, list[Any]]:
    collectors = {name: Collector() for name in tools}
    streamer = ToolArgStreamer(dict(collectors), _tool_call_id)

    for index, event in events:
        if isinstance(event, str):
            streamer.identify_tool(index, event)
        else:
            streamer.on_argument(index, event)
    await streamer.close()
    return {name: collector.events for name, collector in collectors.items()}


def _delta_text(events: list[Any]) -> str:
    return "".join(e.text for e in events if isinstance(e, StringDelta))


# --- router unit tests --------------------------------------------------------


async def test_streams_string_argument():
    collector = await run_router(
        [
            (0, "ask_human"),
            (0, StringDelta(name="question", text="Hello ")),
            (0, StringDelta(name="question", text="world")),
            (0, StringEnd(name="question", value="Hello world")),
        ]
    )
    events = collector.events

    assert _delta_text(events) == "Hello world"
    assert all(
        isinstance(e, StringDelta) and e.name == "question"
        for e in events
        if isinstance(e, StringDelta)
    )
    assert events[-1] == StringEnd(name="question", value="Hello world")


async def test_streams_scalar_arguments_whole():
    collector = await run_router(
        [
            (0, "ask_human"),
            (0, Scalar(name="count", value=42)),
            (0, Scalar(name="ok", value=True)),
            (0, Scalar(name="note", value=None)),
        ]
    )
    events = collector.events

    assert Scalar(name="count", value=42) in events
    assert Scalar(name="ok", value=True) in events
    assert Scalar(name="note", value=None) in events
    assert not any(isinstance(e, StringDelta) for e in events)


async def test_resolves_when_arguments_precede_name():
    collector = await run_router(
        [
            (0, StringDelta(name="question", text="Hi there")),
            (0, StringEnd(name="question", value="Hi there")),
            (0, "ask_human"),
        ]
    )
    events = collector.events

    assert _delta_text(events) == "Hi there"
    assert StringEnd(name="question", value="Hi there") in events


async def test_unregistered_tool_is_ignored():
    collector = await run_router(
        [
            (0, "other"),
            (0, StringEnd(name="question", value="Hi")),
        ]
    )

    assert collector.events == []


async def test_duplicate_tool_name_resolution_is_ignored():
    collector = await run_router(
        [
            (0, "ask_human"),
            (0, "ask_human"),
            (0, StringEnd(name="question", value="Hi")),
        ]
    )

    assert collector.tool_call_ids == ["call-0"]
    assert collector.events == [StringEnd(name="question", value="Hi")]


async def test_routes_multiple_tool_calls_independently():
    events = await run_router_handlers(
        [
            (0, "ask_a"),
            (1, "ask_b"),
            (0, StringDelta(name="question", text="A?")),
            (0, StringEnd(name="question", value="A?")),
            (1, StringDelta(name="question", text="B?")),
            (1, StringEnd(name="question", value="B?")),
        ],
        ["ask_a", "ask_b"],
    )

    assert StringEnd(name="question", value="A?") in events["ask_a"]
    assert StringEnd(name="question", value="B?") in events["ask_b"]
    # No cross-talk between the two calls' streams.
    assert _delta_text(events["ask_a"]) == "A?"
    assert _delta_text(events["ask_b"]) == "B?"


async def test_routes_multiple_arguments_of_one_call():
    events = (
        await run_router_handlers(
            [
                (0, "ask"),
                (0, StringDelta(name="question", text="Hi")),
                (0, StringEnd(name="question", value="Hi")),
                (0, Scalar(name="count", value=3)),
            ],
            ["ask"],
        )
    )["ask"]

    # A single multiplexed stream carries both arguments, distinguished by name.
    assert _delta_text(events) == "Hi"
    assert StringEnd(name="question", value="Hi") in events
    assert Scalar(name="count", value=3) in events
