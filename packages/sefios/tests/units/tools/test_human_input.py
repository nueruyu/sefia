from sefia.llm._arg_stream import _ArgStreamChannel
from sefia.streaming import StringDelta
from sefia.tool_collectors import DefaultToolCollector
from sefios.tools import HUMAN_INPUT_TOOL_NAME, HumanInputTool


class Agent:
    """A tool's own methods are never self-exposed; hold it as a dependency."""

    def __init__(self, human_input: HumanInputTool):
        self._human_input = human_input


async def test_human_input_tool_streams_question_deltas():
    seen: list[str] = []
    agent = Agent(HumanInputTool(on_question_delta=seen.append))
    registry = DefaultToolCollector().collect(agent)
    registered = next(
        tool for tool in registry.get_all() if "get_human_input" in tool.name
    )
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="question", text="What "))
    channel.feed(StringDelta(name="question", text="topic?"))
    channel.feed(StringDelta(name="other", text="ignored"))
    channel.close()

    await registered.stream_handler(channel)

    assert seen == ["What ", "topic?"]


def test_human_input_tool_name_marker_matches_collected_tool():
    agent = Agent(HumanInputTool())
    registry = DefaultToolCollector().collect(agent)

    assert HUMAN_INPUT_TOOL_NAME in registry.get_names()
