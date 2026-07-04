from typing import Protocol

from sefia import stream_for
from sefia.llm._arg_stream import _ArgStreamChannel
from sefia.streaming import ArgStream, StringDelta
from sefia.tool_collectors import DefaultToolCollector


async def test_stream_handler_is_collected_and_bound_to_instance():
    seen_self = []

    class Toolkit:
        async def ask_human(self, question: str) -> str:
            return question

        @stream_for(ask_human)
        async def _ask_human_stream(self, events) -> None:
            seen_self.append(self)
            async for _ in events:
                pass

    class Agent:
        def __init__(self):
            self._toolkit = Toolkit()

    agent = Agent()
    registry = DefaultToolCollector().collect(agent)

    (registered,) = registry.get_all()
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.close()
    await registered.stream_handler(channel)
    assert seen_self == [agent._toolkit]  # bound to the toolkit instance


async def test_bound_stream_handler_consumes_events():
    received = []

    class Toolkit:
        async def ask_human(self, question: str) -> str:
            return question

        @stream_for(ask_human)
        async def _ask_human_stream(self, events) -> None:
            async for event in events:
                received.append(event)

    class Agent:
        def __init__(self):
            self._toolkit = Toolkit()

    registry = DefaultToolCollector().collect(Agent())
    (registered,) = registry.get_all()
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="question", text="hi"))
    channel.close()
    await registered.stream_handler(channel)

    assert received == [StringDelta(name="question", text="hi")]


def test_tool_without_stream_handler_has_none():
    class Toolkit:
        async def plain(self, x: str) -> str:
            return x

    class Agent:
        def __init__(self):
            self._toolkit = Toolkit()

    registry = DefaultToolCollector().collect(Agent())

    (registered,) = registry.get_all()
    assert registered.stream_handler is None


async def test_static_tool_stream_handler_is_collected():
    received = []

    class Toolkit:
        @staticmethod
        async def ask_human(question: str) -> str:
            return question

        @stream_for(ask_human)
        async def _ask_human_stream(events: ArgStream) -> None:
            async for event in events:
                received.append(event)

    class Agent:
        def __init__(self):
            self._toolkit = Toolkit()

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

    class Toolkit:
        @classmethod
        async def ask_human(cls, question: str) -> str:
            return question

        @stream_for(ask_human)
        async def _ask_human_stream(cls, events: ArgStream) -> None:
            seen_cls.append(cls)
            async for _ in events:
                pass

    class Agent:
        def __init__(self):
            self._toolkit = Toolkit()

    registry = DefaultToolCollector().collect(Agent())
    registered = next(tool for tool in registry.get_all() if "ask_human" in tool.name)
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.close()
    await registered.stream_handler(channel)

    assert seen_cls == [Toolkit]


async def test_stream_handler_is_found_when_the_field_is_protocol_narrowed():
    # stream_for is applied to the implementation's method, which is a
    # different function object than the Protocol's own declared method — the
    # handler lookup must not be tied to whichever one supplied the schema.
    received = []

    class AskHuman(Protocol):
        async def ask_human(self, question: str) -> str: ...

    class Toolkit:
        async def ask_human(self, question: str) -> str:
            return question

        @stream_for(ask_human)
        async def _ask_human_stream(self, events) -> None:
            async for event in events:
                received.append(event)

    class Agent:
        _toolkit: AskHuman

        def __init__(self):
            self._toolkit = Toolkit()

    registry = DefaultToolCollector().collect(Agent())
    (registered,) = registry.get_all()
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="question", text="hi"))
    channel.close()
    await registered.stream_handler(channel)

    assert received == [StringDelta(name="question", text="hi")]
