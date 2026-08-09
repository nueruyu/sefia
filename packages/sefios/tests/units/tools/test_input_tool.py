import pytest

from sefia import Tools
from sefia.inference import Capability
from sefia.llm._arg_stream import _ArgStreamChannel
from sefia.streaming import StringDelta
from sefia.testing import MockLLMClient, memory_session
from sefia.tool_collectors import DefaultToolCollector
from sefios.tools import Input


class Agent:
    """A tool's own methods are never self-exposed; hold it as a dependency."""

    _input: Tools[Input]

    def __init__(self, input_tool: Input):
        self._input = input_tool


async def test_input_tool_streams_prompt_deltas():
    seen: list[tuple[str, str]] = []
    agent = Agent(
        Input(on_prompt_delta=lambda call_id, text: seen.append((call_id, text)))
    )
    registry = DefaultToolCollector().collect([Capability(value=agent, declared=None)])
    registered = next(tool for tool in registry.get_all() if "get_input" in tool.name)
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="prompt", text="What "))
    channel.feed(StringDelta(name="prompt", text="topic?"))
    channel.feed(StringDelta(name="other", text="ignored"))
    channel.close()

    await registered.stream_handler("call-1", channel)

    assert seen == [("call-1", "What "), ("call-1", "topic?")]


async def test_input_fails_fast_outside_tool_dispatch():
    async with memory_session(MockLLMClient([])):
        with pytest.raises(RuntimeError, match="only available inside a tool call"):
            await Input().get_input("Name?")
