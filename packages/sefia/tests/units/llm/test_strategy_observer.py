from unittest.mock import AsyncMock, Mock

import pytest
from sefia.event_system import EventPublisher
from sefia.llm._arg_stream import ToolArgStreamer
from sefia.llm._strategy import _StrategyDecisionObserver
from sefia.llm.events import BeforeLLMCall, LLMReasoningTokenReceived, LLMTokenReceived
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.streaming import OutputStreamEvent, Scalar, StringDelta, StringEnd
from sefia.streaming import (
    ArgEvent,
)
from sefia.streaming import (
    Scalar as ArgScalar,
)
from sefia.streaming import (
    StringDelta as ArgStringDelta,
)
from sefia.streaming import (
    StringEnd as ArgStringEnd,
)


async def test_observer_publishes_prompt_and_tokens() -> None:
    publisher = AsyncMock(spec=EventPublisher)
    spec = Mock(spec=DecisionSpec)
    observer = _StrategyDecisionObserver(publisher, spec, None)

    await observer.before_request("prompt")
    await observer.response_text("token")
    await observer.reasoning_text("thinking")

    assert [c.args[0] for c in publisher.publish.await_args_list] == [
        BeforeLLMCall(prompt="prompt", decision_spec=spec),
        LLMTokenReceived(token="token"),
        LLMReasoningTokenReceived(token="thinking"),
    ]


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            StringDelta(("tool_calls", 2, "arguments", "query"), "he"),
            ArgStringDelta(name="query", text="he"),
        ),
        (
            StringEnd(("tool_calls", 2, "arguments", "query"), "hello"),
            ArgStringEnd(name="query", value="hello"),
        ),
        (
            Scalar(("tool_calls", 2, "arguments", "limit"), 3),
            ArgScalar(name="limit", value=3),
        ),
    ],
)
async def test_observer_converts_argument_events(
    event: OutputStreamEvent, expected: ArgEvent
) -> None:
    streamer = Mock(spec=ToolArgStreamer)
    observer = _StrategyDecisionObserver(
        AsyncMock(spec=EventPublisher), Mock(spec=DecisionSpec), streamer
    )

    await observer.output(event)

    streamer.on_argument.assert_called_once_with(2, expected)
    streamer.identify_tool.assert_not_called()


async def test_observer_identifies_tool_from_completed_name() -> None:
    streamer = Mock(spec=ToolArgStreamer)
    observer = _StrategyDecisionObserver(
        AsyncMock(spec=EventPublisher), Mock(spec=DecisionSpec), streamer
    )

    await observer.output(StringEnd(("tool_calls", 2, "name"), "lookup"))

    streamer.identify_tool.assert_called_once_with(2, "lookup")
    streamer.on_argument.assert_not_called()


@pytest.mark.parametrize(
    "event",
    [
        StringDelta(("tool_calls", 0, "name"), "look"),
        StringEnd(("result",), "done"),
        Scalar(("tool_calls", 0, "arguments", "nested", "value"), 1),
    ],
)
async def test_observer_ignores_unrelated_paths(event: OutputStreamEvent) -> None:
    streamer = Mock(spec=ToolArgStreamer)
    observer = _StrategyDecisionObserver(
        AsyncMock(spec=EventPublisher), Mock(spec=DecisionSpec), streamer
    )
    await observer.output(event)
    assert streamer.mock_calls == []


async def test_observer_ignores_output_without_a_streamer() -> None:
    publisher = AsyncMock(spec=EventPublisher)
    observer = _StrategyDecisionObserver(publisher, Mock(spec=DecisionSpec), None)
    await observer.output(StringEnd(("tool_calls", 0, "name"), "lookup"))
    publisher.publish.assert_not_called()
