from sefia.streaming import StringDelta, _ArgStreamChannel
from sefia.tool_collectors import DefaultToolCollector
from sefios.tools import HumanInputTool


async def test_human_input_tool_streams_question_deltas():
    seen: list[str] = []
    tool = HumanInputTool(on_question_delta=seen.append)
    registry = DefaultToolCollector().collect(tool)
    registered = next(
        tool
        for tool in registry.get_all()
        if "get_human_input" in tool.name
    )
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="question", text="What "))
    channel.feed(StringDelta(name="question", text="topic?"))
    channel.feed(StringDelta(name="other", text="ignored"))
    channel.close()

    await registered.stream_handler(channel)

    assert seen == ["What ", "topic?"]
