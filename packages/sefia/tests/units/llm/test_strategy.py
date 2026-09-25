from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture
from sefia import ToolRegistry
from sefia.event_system import EventPublisher
from sefia.exceptions import InvalidInferenceResponseError, UnknownToolDecisionError
from sefia.inference import (
    ResultDecision,
    ToolCallResult,
    ToolCallsDecision,
)
from sefia.llm import LLMCompletion, LLMInferenceStrategy, Message
from sefia.llm.events import (
    AfterLLMCall,
    BeforeLLMCall,
    LLMReasoningTokenReceived,
    LLMTokenReceived,
)
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import (
    DecisionObserver,
    DecisionToolCalls,
    DecisionToolResult,
    DecodedDecision,
)
from sefia.testing import make_function_info, make_tool_call_request


class _ArgumentModel(BaseModel):
    value: int


@dataclass(frozen=True)
class _Record:
    identifier: UUID
    created_at: datetime


@pytest.mark.parametrize("stream", [False, True])
async def test_strategy_passes_request_to_transport_and_validates_result(
    transport: AsyncMock,
    make_strategy: Callable[..., LLMInferenceStrategy],
    stream: bool,
) -> None:
    strategy = make_strategy(stream=stream)
    function = make_function_info(instructions="do it", return_type=str)
    publisher = AsyncMock(spec=EventPublisher)

    decision = await strategy.decide_next_step(function, [], ToolRegistry(), publisher)

    assert isinstance(decision, ResultDecision)
    assert decision.result == "done"
    transport.request_decision.assert_awaited_once()
    sent = transport.request_decision.await_args.kwargs
    assert sent["client"] is strategy.llm_client
    assert sent["stream"] is stream
    assert sent["request"].function is function
    assert sent["request"].history == ()
    assert sent["request"].rejected is None
    assert sent["request"].decision_spec.result is not None
    publisher.publish.assert_awaited_once_with(
        AfterLLMCall(transport.request_decision.return_value.completion)
    )
    assert "dump" not in sent


async def test_strategy_materializes_arguments_and_history_before_transport(
    transport: AsyncMock,
    make_strategy: Callable[..., LLMInferenceStrategy],
) -> None:
    identifier = UUID("12345678-1234-5678-1234-567812345678")
    created_at = datetime(2026, 9, 23, 10, 30)
    record = _Record(identifier=identifier, created_at=created_at)
    model = _ArgumentModel(value=3)
    function = make_function_info(
        bound_arguments={"model": model, "record": record},
        return_type=str,
    )
    history = [
        ToolCallsDecision(
            [
                make_tool_call_request(
                    id="call-1",
                    name="lookup",
                    arguments={"record": record},
                )
            ]
        ),
        ToolCallResult(tool_call_id="call-1", result=model),
    ]

    await make_strategy().decide_next_step(
        function,
        history,
        ToolRegistry(),
        AsyncMock(spec=EventPublisher),
    )

    request = transport.request_decision.await_args.kwargs["request"]
    assert request.arguments.tree == {
        "model": {"value": 3},
        "record": {
            "identifier": str(identifier),
            "created_at": "2026-09-23T10:30:00",
        },
    }
    tool_calls, tool_result = request.history
    assert isinstance(tool_calls, DecisionToolCalls)
    assert tool_calls.calls[0].arguments.tree == {
        "record": {
            "identifier": str(identifier),
            "created_at": "2026-09-23T10:30:00",
        }
    }
    assert isinstance(tool_result, DecisionToolResult)
    assert tool_result.result.tree == {"value": 3}


async def test_strategy_assigns_ids_to_validated_tool_calls(
    transport: AsyncMock, make_strategy: Callable[..., LLMInferenceStrategy]
) -> None:
    registry = ToolRegistry()

    def my_tool(param: int) -> int:
        return param

    registry.add(my_tool, name="my_tool")
    data = StructuredData.from_json(
        {
            "decision": "tool_calls",
            "tool_calls": [{"name": "my_tool", "arguments": {"param": 1}}],
        }
    )
    transport.request_decision.return_value = DecodedDecision(data, LLMCompletion())

    decision = await make_strategy().decide_next_step(
        make_function_info(return_type=str),
        [],
        registry,
        AsyncMock(spec=EventPublisher),
    )

    assert isinstance(decision, ToolCallsDecision)
    assert decision.calls[0].name == "my_tool"
    assert decision.calls[0].arguments == {"param": 1}
    assert decision.calls[0].id.startswith("call_")


async def test_unknown_tool_preserves_specific_cause(
    transport: AsyncMock, make_strategy: Callable[..., LLMInferenceStrategy]
) -> None:
    registry = ToolRegistry()
    registry.add(lambda: None, name="known")
    data = StructuredData.from_json(
        {"decision": "tool_calls", "tool_calls": [{"name": "unknown", "arguments": {}}]}
    )
    transport.request_decision.return_value = DecodedDecision(
        data, LLMCompletion(content="invalid")
    )

    with pytest.raises(InvalidInferenceResponseError) as exc_info:
        await make_strategy(max_repair_attempts=0).decide_next_step(
            make_function_info(return_type=str),
            [],
            registry,
            AsyncMock(spec=EventPublisher),
        )

    assert isinstance(exc_info.value.__cause__, UnknownToolDecisionError)
    assert exc_info.value.__cause__.tool_name == "unknown"
    assert exc_info.value.raw_content == "invalid"


@pytest.mark.parametrize("fails", [False, True])
async def test_argument_streamer_is_closed_after_transport(
    transport: AsyncMock,
    make_strategy: Callable[..., LLMInferenceStrategy],
    mocker: MockerFixture,
    fails: bool,
) -> None:
    streamer = mocker.patch(
        "sefia.llm._strategy.ToolArgStreamer", autospec=True
    ).return_value
    registry = ToolRegistry()
    registry.add(lambda: None, name="tool", stream_handler=AsyncMock())
    if fails:
        transport.request_decision.side_effect = RuntimeError("transport failed")
    strategy = make_strategy(stream=True)

    if fails:
        with pytest.raises(RuntimeError, match="transport failed"):
            await strategy.decide_next_step(
                make_function_info(return_type=str),
                [],
                registry,
                AsyncMock(spec=EventPublisher),
            )
    else:
        await strategy.decide_next_step(
            make_function_info(return_type=str),
            [],
            registry,
            AsyncMock(spec=EventPublisher),
        )

    streamer.close.assert_awaited_once()


async def test_transport_observer_notifies_the_supplied_publisher(
    transport: AsyncMock, make_strategy: Callable[..., LLMInferenceStrategy]
) -> None:
    publisher = AsyncMock(spec=EventPublisher)
    decoded = transport.request_decision.return_value

    async def respond(
        *, observer: DecisionObserver, **kwargs: object
    ) -> DecodedDecision:
        await observer.before_request((Message(role="user", content="prompt"),))
        await observer.response_text("answer")
        await observer.reasoning_text("thinking")
        return decoded

    transport.request_decision.side_effect = respond
    await make_strategy(stream=True).decide_next_step(
        make_function_info(return_type=str), [], ToolRegistry(), publisher
    )

    spec = transport.request_decision.await_args.kwargs["request"].decision_spec
    assert [c.args[0] for c in publisher.publish.await_args_list] == [
        BeforeLLMCall(
            messages=(Message(role="user", content="prompt"),), decision_spec=spec
        ),
        LLMTokenReceived(token="answer"),
        LLMReasoningTokenReceived(token="thinking"),
        AfterLLMCall(decoded.completion),
    ]
