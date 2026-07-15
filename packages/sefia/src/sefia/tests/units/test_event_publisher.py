from dataclasses import dataclass
from typing import Generic, TypeVar, Union

import pytest
from sefia.exceptions import PauseException
from pytest_mock import MockerFixture

from sefia.event_system import EventHandler, EventPublisher
from sefia.events import AfterToolCall, BeforeToolCall, Event, ToolExecutionFailed
from sefia.inference import ToolCallRequest

T = TypeVar("T")


@dataclass(frozen=True)
class GenericEvent(Event, Generic[T]):
    value: T


class MyEventHandler(EventHandler[BeforeToolCall]):
    async def handle(self, event: BeforeToolCall) -> None:
        pass


class MyUnionEventHandler(EventHandler[Union[AfterToolCall, ToolExecutionFailed]]):
    async def handle(self, event: Union[AfterToolCall, ToolExecutionFailed]) -> None:
        pass


class MyPipeUnionEventHandler(EventHandler[AfterToolCall | ToolExecutionFailed]):
    async def handle(self, event: AfterToolCall | ToolExecutionFailed) -> None:
        pass


class MyBaseEventHandler(EventHandler[Event]):
    async def handle(self, event: Event) -> None:
        pass


class MyGenericEventHandler(EventHandler[GenericEvent[int]]):
    async def handle(self, event: GenericEvent[int]) -> None:
        pass


class AfterToolCallHandler(EventHandler[AfterToolCall]):
    async def handle(self, event: AfterToolCall) -> None:
        pass


class ToolExecutionFailedHandler(EventHandler[ToolExecutionFailed]):
    async def handle(self, event: ToolExecutionFailed) -> None:
        pass


async def _handle_multi_inherited_event(
    self, event: AfterToolCall | ToolExecutionFailed
) -> None:
    pass


MultiInheritedEventHandler = type(
    "MultiInheritedEventHandler",
    (AfterToolCallHandler, ToolExecutionFailedHandler),
    {"handle": _handle_multi_inherited_event},
)


@pytest.fixture
def before_tool_call_event() -> BeforeToolCall:
    return BeforeToolCall(tool_call=ToolCallRequest(id="1", name="test", arguments={}))


class TestEventPublisher:
    async def test_dispatches_event_to_correct_handler(
        self, mocker: MockerFixture, before_tool_call_event: BeforeToolCall
    ):
        handler = MyEventHandler()
        spy = mocker.spy(handler, "handle")
        publisher = EventPublisher(handlers=[handler])

        await publisher.publish(before_tool_call_event)

        spy.assert_called_once_with(before_tool_call_event)

    async def test_dispatches_event_to_union_handler(self, mocker: MockerFixture):
        handler = MyUnionEventHandler()
        spy = mocker.spy(handler, "handle")
        publisher = EventPublisher(handlers=[handler])
        event1 = AfterToolCall(
            tool_call=ToolCallRequest(id="1", name="test", arguments={}), result="ok"
        )
        event2 = ToolExecutionFailed(
            tool_call=ToolCallRequest(id="2", name="test", arguments={}),
            error=ValueError(),
        )

        await publisher.publish(event1)
        await publisher.publish(event2)

        assert spy.call_count == 2
        spy.assert_any_call(event1)
        spy.assert_any_call(event2)

    async def test_dispatches_event_to_pipe_union_handler(self, mocker: MockerFixture):
        handler = MyPipeUnionEventHandler()
        spy = mocker.spy(handler, "handle")
        publisher = EventPublisher(handlers=[handler])
        event1 = AfterToolCall(
            tool_call=ToolCallRequest(id="1", name="test", arguments={}), result="ok"
        )
        event2 = ToolExecutionFailed(
            tool_call=ToolCallRequest(id="2", name="test", arguments={}),
            error=ValueError(),
        )

        await publisher.publish(event1)
        await publisher.publish(event2)

        assert spy.call_count == 2
        spy.assert_any_call(event1)
        spy.assert_any_call(event2)

    async def test_dispatches_event_to_generic_handler(self, mocker: MockerFixture):
        handler = MyGenericEventHandler()
        spy = mocker.spy(handler, "handle")
        publisher = EventPublisher(handlers=[handler])
        event = GenericEvent[int](value=1)

        await publisher.publish(event)

        spy.assert_called_once_with(event)

    async def test_dispatches_event_to_multi_inherited_handler(
        self, mocker: MockerFixture
    ):
        handler = MultiInheritedEventHandler()
        spy = mocker.spy(handler, "handle")
        publisher = EventPublisher(handlers=[handler])
        event1 = AfterToolCall(
            tool_call=ToolCallRequest(id="1", name="test", arguments={}), result="ok"
        )
        event2 = ToolExecutionFailed(
            tool_call=ToolCallRequest(id="2", name="test", arguments={}),
            error=ValueError(),
        )

        await publisher.publish(event1)
        await publisher.publish(event2)

        assert spy.call_count == 2
        spy.assert_any_call(event1)
        spy.assert_any_call(event2)

    async def test_does_not_dispatch_to_wrong_handler(
        self, mocker: MockerFixture, before_tool_call_event: BeforeToolCall
    ):
        handler = MyUnionEventHandler()  # Listens for AfterToolCall, ToolError
        spy = mocker.spy(handler, "handle")
        publisher = EventPublisher(handlers=[handler])

        await publisher.publish(before_tool_call_event)  # This is a BeforeToolCall

        spy.assert_not_called()

    async def test_dispatches_to_multiple_handlers(
        self, mocker: MockerFixture, before_tool_call_event: BeforeToolCall
    ):
        handler1 = MyEventHandler()
        handler2 = MyEventHandler()
        spy1 = mocker.spy(handler1, "handle")
        spy2 = mocker.spy(handler2, "handle")
        publisher = EventPublisher(handlers=[handler1, handler2])

        await publisher.publish(before_tool_call_event)

        spy1.assert_called_once_with(before_tool_call_event)
        spy2.assert_called_once_with(before_tool_call_event)

    async def test_dispatches_to_base_event_handler(
        self, mocker: MockerFixture, before_tool_call_event: BeforeToolCall
    ):
        handler = MyBaseEventHandler()
        spy = mocker.spy(handler, "handle")
        publisher = EventPublisher(handlers=[handler])

        await publisher.publish(before_tool_call_event)

        spy.assert_called_once_with(before_tool_call_event)

    async def test_isolates_handler_exceptions(
        self, mocker: MockerFixture, before_tool_call_event: BeforeToolCall
    ):
        # Observation handlers must not break the core loop: a handler that
        # raises is logged and swallowed, and later handlers still run.
        class RaisingHandler(EventHandler[BeforeToolCall]):
            async def handle(self, event: BeforeToolCall) -> None:
                raise ValueError("handler is broken")

        good_handler = MyEventHandler()
        spy = mocker.spy(good_handler, "handle")
        publisher = EventPublisher(handlers=[RaisingHandler(), good_handler])

        # Should not raise.
        await publisher.publish(before_tool_call_event)

        spy.assert_called_once_with(before_tool_call_event)

    async def test_yield_exception_is_swallowed(
        self, mocker: MockerFixture, before_tool_call_event: BeforeToolCall
    ):
        # Observers cannot steer control flow: even a PauseException raised by a
        # handler is logged and swallowed, never leaked out of publish(). Genuine
        # resumable interrupts come from the control layer (e.g. tools), not
        # observers. Later handlers still run.
        class YieldingHandler(EventHandler[BeforeToolCall]):
            async def handle(self, event: BeforeToolCall) -> None:
                raise PauseException("resume later")

        good_handler = MyEventHandler()
        spy = mocker.spy(good_handler, "handle")
        publisher = EventPublisher(handlers=[YieldingHandler(), good_handler])

        # Should not raise.
        await publisher.publish(before_tool_call_event)

        spy.assert_called_once_with(before_tool_call_event)
