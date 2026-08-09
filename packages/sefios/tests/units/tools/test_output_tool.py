import pytest

from sefia import Tools
from sefia.inference import Capability
from sefia.llm._arg_stream import _ArgStreamChannel
from sefia.streaming import StringDelta
from sefia.testing import MockLLMClient, memory_session
from sefia.tool_collectors import DefaultToolCollector
from sefios.tools import Output


class Agent:
    """A tool's own methods are never self-exposed; hold it as a dependency."""

    _output: Tools[Output]

    def __init__(self, output_tool: Output):
        self._output = output_tool


async def test_output_tool_streams_message_deltas():
    seen: list[tuple[str, str]] = []
    agent = Agent(
        Output(on_message_delta=lambda call_id, text: seen.append((call_id, text)))
    )
    registry = DefaultToolCollector().collect([Capability(value=agent, declared=None)])
    registered = next(tool for tool in registry.get_all() if "send_output" in tool.name)
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="message", text="Here "))
    channel.feed(StringDelta(name="message", text="you go."))
    channel.feed(StringDelta(name="other", text="ignored"))
    channel.close()

    await registered.stream_handler("call-1", channel)

    assert seen == [("call-1", "Here "), ("call-1", "you go.")]


async def test_output_fails_fast_outside_tool_dispatch():
    async with memory_session(MockLLMClient([])):
        with pytest.raises(RuntimeError, match="only available inside a tool call"):
            await Output().send_output("message")
