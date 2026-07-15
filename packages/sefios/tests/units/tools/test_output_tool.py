from sefia.llm._arg_stream import _ArgStreamChannel
from sefia.streaming import StringDelta
from sefia.tool_collectors import DefaultToolCollector
from sefios.tools import OutputTool


class Agent:
    """A tool's own methods are never self-exposed; hold it as a dependency."""

    def __init__(self, output_tool: OutputTool):
        self._output = output_tool


async def test_output_tool_streams_message_deltas():
    seen: list[str] = []
    agent = Agent(OutputTool(on_message_delta=seen.append))
    registry = DefaultToolCollector().collect(agent)
    registered = next(tool for tool in registry.get_all() if "send_output" in tool.name)
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="message", text="Here "))
    channel.feed(StringDelta(name="message", text="you go."))
    channel.feed(StringDelta(name="other", text="ignored"))
    channel.close()

    await registered.stream_handler(channel)

    assert seen == ["Here ", "you go."]
