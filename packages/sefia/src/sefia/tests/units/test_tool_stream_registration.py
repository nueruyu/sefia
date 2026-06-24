from sefia import tool
from sefia.tool_collectors import DefaultToolCollector
from sefia.streaming import StringDelta, _ArgStreamChannel


async def test_stream_handler_is_collected_and_bound_to_instance():
    seen_self = []

    class Agent:
        @tool
        async def ask_human(self, question: str) -> str:
            return question

        @ask_human.stream
        async def _ask_human_stream(self, events) -> None:
            seen_self.append(self)
            async for _ in events:
                pass

    agent = Agent()
    registry = DefaultToolCollector().collect(agent)

    (registered,) = registry.get_all()
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.close()
    await registered.stream_handler(channel)
    assert seen_self == [agent]  # the handler was bound to this instance


async def test_bound_stream_handler_consumes_events():
    received = []

    class Agent:
        @tool
        async def ask_human(self, question: str) -> str:
            return question

        @ask_human.stream
        async def _ask_human_stream(self, events) -> None:
            async for event in events:
                received.append(event)

    agent = Agent()
    registry = DefaultToolCollector().collect(agent)
    (registered,) = registry.get_all()
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="question", text="hi"))
    channel.close()
    await registered.stream_handler(channel)

    assert received == [StringDelta(name="question", text="hi")]


def test_tool_without_stream_handler_has_none():
    class Agent:
        @tool
        async def plain(self, x: str) -> str:
            return x

    registry = DefaultToolCollector().collect(Agent())

    (registered,) = registry.get_all()
    assert registered.stream_handler is None
