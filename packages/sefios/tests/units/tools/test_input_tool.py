from sefia import Tools
from sefia.inference import Capability
from sefia.llm._arg_stream import _ArgStreamChannel
from sefia.streaming import StringDelta
from sefia.tool_collectors import DefaultToolCollector
from sefios.tools import InputTool


class Agent:
    """A tool's own methods are never self-exposed; hold it as a dependency."""

    _input: Tools[InputTool]

    def __init__(self, input_tool: InputTool):
        self._input = input_tool


async def test_input_tool_streams_prompt_deltas():
    seen: list[str] = []
    agent = Agent(InputTool(on_prompt_delta=seen.append))
    registry = DefaultToolCollector().collect([Capability(value=agent, declared=None)])
    registered = next(tool for tool in registry.get_all() if "get_input" in tool.name)
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="prompt", text="What "))
    channel.feed(StringDelta(name="prompt", text="topic?"))
    channel.feed(StringDelta(name="other", text="ignored"))
    channel.close()

    await registered.stream_handler(channel)

    assert seen == ["What ", "topic?"]
