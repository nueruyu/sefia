import pytest

from sefia import ToolRegistry, Tools
from sefia._tool_execution import call_tools
from sefia.event_system import EventPublisher
from sefia.inference import Capability, ToolCallRequest
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
        with pytest.raises(RuntimeError, match="must be invoked as a dispatched tool"):
            await Output().send_output("message")


async def test_nested_output_fails_instead_of_reusing_parent_call_id():
    seen = []
    output = Output(on_output=seen.append)

    async def parent() -> str:
        return await output.send_output("message")

    registry = ToolRegistry()
    registry.add(parent, name="parent")

    async with memory_session(MockLLMClient([])):
        results = await call_tools(
            [ToolCallRequest(id="parent-call", name="parent", arguments={})],
            registry,
            EventPublisher([]),
        )

    assert results[0].tool_call_id == "parent-call"
    assert (
        "Output.send_output() must be invoked as a dispatched tool" in results[0].result
    )
    assert seen == []
