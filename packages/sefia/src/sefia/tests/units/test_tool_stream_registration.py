from sefia import tool, toolify
from sefia.llm._arg_stream import _ArgStreamChannel
from sefia.streaming import ArgStream, StringDelta
from sefia.tool_collectors import DefaultToolCollector


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


async def test_static_tool_stream_handler_is_collected():
    received = []

    class Agent:
        @tool
        @staticmethod
        async def ask_human(question: str) -> str:
            return question

        @ask_human.stream
        async def _ask_human_stream(events: ArgStream) -> None:
            async for event in events:
                received.append(event)

    registry = DefaultToolCollector().collect(Agent())
    registered = next(tool for tool in registry.get_all() if "ask_human" in tool.name)
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="question", text="hi"))
    channel.close()
    await registered.stream_handler(channel)

    assert received == [StringDelta(name="question", text="hi")]


async def test_class_tool_stream_handler_is_bound_to_class():
    seen_cls = []

    class Agent:
        @tool
        @classmethod
        async def ask_human(cls, question: str) -> str:
            return question

        @ask_human.stream
        async def _ask_human_stream(cls, events: ArgStream) -> None:
            seen_cls.append(cls)
            async for _ in events:
                pass

    registry = DefaultToolCollector().collect(Agent())
    registered = next(tool for tool in registry.get_all() if "ask_human" in tool.name)
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.close()
    await registered.stream_handler(channel)

    assert seen_cls == [Agent]


async def test_toolified_standalone_stream_handler_is_collected():
    received = []

    @tool
    async def ask_human(question: str) -> str:
        return question

    @ask_human.stream
    async def _ask_human_stream(events: ArgStream) -> None:
        async for event in events:
            received.append(event)

    class Agent:
        def __init__(self) -> None:
            self._tools = toolify(ask_human)

    registry = DefaultToolCollector().collect(Agent())
    registered = next(tool for tool in registry.get_all() if "ask_human" in tool.name)
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="question", text="hi"))
    channel.close()
    await registered.stream_handler(channel)

    assert received == [StringDelta(name="question", text="hi")]
