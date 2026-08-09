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


async def test_nested_output_does_not_reuse_parent_tool_call_id():
    seen = []
    output = Output(on_output=seen.append)

    async def parent() -> str:
        await output.send_output("first")
        await output.send_output("second")
        return "ok"

    registry = ToolRegistry()
    registry.add(parent, name="parent")

    async with memory_session(MockLLMClient([])):
        await call_tools(
            [ToolCallRequest(id="parent-call", name="parent", arguments={})],
            registry,
            EventPublisher([]),
        )

    ids = [message.interaction_id for message in seen]
    assert len(ids) == 2
    assert "parent-call" not in ids
    assert ids[0] != ids[1]
