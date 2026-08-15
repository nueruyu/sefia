from typing import Protocol

from sefia import Tools, preview
from sefia._tool_system import ToolRegistry
from sefia.inference import Capability
from sefia.llm._arg_stream import _ArgStreamChannel
from sefia.streaming import ArgEvent, ArgStream, StringDelta
from sefia.tool_collectors import DefaultToolCollector


def _collect(instance: object) -> ToolRegistry:
    """Collect tools for ``instance`` as its @infer method's ``self`` capability."""
    return DefaultToolCollector().collect([Capability(value=instance, declared=None)])


async def test_stream_handler_is_collected_and_bound_to_instance() -> None:
    seen_self: list[object] = []

    class Toolkit:
        async def ask_human(self, question: str) -> str:
            return question

        @preview(ask_human)
        async def _ask_human_stream(self, tool_call_id: str, events: ArgStream) -> None:
            seen_self.append(self)
            async for _ in events:
                pass

    class Agent:
        _toolkit: Tools[Toolkit]

        def __init__(self) -> None:
            self._toolkit = Toolkit()

    agent = Agent()
    registry = _collect(agent)

    (registered,) = registry.get_all()
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.close()
    await registered.stream_handler("call-1", channel)
    assert seen_self == [agent._toolkit]  # bound to the toolkit instance


async def test_bound_stream_handler_consumes_events() -> None:
    received: list[ArgEvent] = []

    class Toolkit:
        async def ask_human(self, question: str) -> str:
            return question

        @preview(ask_human)
        async def _ask_human_stream(self, tool_call_id: str, events: ArgStream) -> None:
            async for event in events:
                received.append(event)

    class Agent:
        _toolkit: Tools[Toolkit]

        def __init__(self) -> None:
            self._toolkit = Toolkit()

    registry = _collect(Agent())
    (registered,) = registry.get_all()
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="question", text="hi"))
    channel.close()
    await registered.stream_handler("call-1", channel)

    assert received == [StringDelta(name="question", text="hi")]


def test_tool_without_stream_handler_has_none() -> None:
    class Toolkit:
        async def plain(self, x: str) -> str:
            return x

    class Agent:
        _toolkit: Tools[Toolkit]

        def __init__(self) -> None:
            self._toolkit = Toolkit()

    registry = _collect(Agent())

    (registered,) = registry.get_all()
    assert registered.stream_handler is None


async def test_static_tool_stream_handler_is_collected() -> None:
    received: list[ArgEvent] = []

    class Toolkit:
        @staticmethod
        async def ask_human(question: str) -> str:
            return question

        @staticmethod
        @preview(ask_human)
        async def _ask_human_stream(tool_call_id: str, events: ArgStream) -> None:
            async for event in events:
                received.append(event)

    class Agent:
        _toolkit: Tools[Toolkit]

        def __init__(self) -> None:
            self._toolkit = Toolkit()

    registry = _collect(Agent())
    registered = next(tool for tool in registry.get_all() if "ask_human" in tool.name)
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="question", text="hi"))
    channel.close()
    await registered.stream_handler("call-1", channel)

    assert received == [StringDelta(name="question", text="hi")]


async def test_class_tool_stream_handler_is_bound_to_class() -> None:
    seen_cls: list[object] = []

    class Toolkit:
        @classmethod
        async def ask_human(cls, question: str) -> str:
            return question

        @preview(ask_human)
        async def _ask_human_stream(cls, tool_call_id: str, events: ArgStream) -> None:
            seen_cls.append(cls)
            async for _ in events:
                pass

    class Agent:
        _toolkit: Tools[Toolkit]

        def __init__(self) -> None:
            self._toolkit = Toolkit()

    registry = _collect(Agent())
    registered = next(tool for tool in registry.get_all() if "ask_human" in tool.name)
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.close()
    await registered.stream_handler("call-1", channel)

    assert seen_cls == [Toolkit]


async def test_stream_handler_is_found_when_the_field_is_protocol_narrowed() -> None:
    # preview is applied to the implementation's method, which is a
    # different function object than the Protocol's own declared method — the
    # handler lookup must not be tied to whichever one supplied the schema.
    received: list[ArgEvent] = []

    class AskHuman(Protocol):
        async def ask_human(self, question: str) -> str: ...

    class Toolkit:
        async def ask_human(self, question: str) -> str:
            return question

        @preview(ask_human)
        async def _ask_human_stream(self, tool_call_id: str, events: ArgStream) -> None:
            async for event in events:
                received.append(event)

    class Agent:
        _toolkit: Tools[AskHuman]

        def __init__(self) -> None:
            self._toolkit = Toolkit()

    registry = _collect(Agent())
    (registered,) = registry.get_all()
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="question", text="hi"))
    channel.close()
    await registered.stream_handler("call-1", channel)

    assert received == [StringDelta(name="question", text="hi")]
