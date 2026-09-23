import json
from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest
from sefia import ToolRegistry
from sefia.event_system import EventPublisher
from sefia.exceptions import InvalidInferenceResponseError
from sefia.inference import ResultDecision, ToolCallResult, ToolCallsDecision
from sefia.llm import LLMCompletion, LLMInferenceStrategy, ToolCall
from sefia.llm.events import DecisionRepairAttempt
from sefia.llm.exceptions import DecisionDecodingError, LLMCompletionDecodingError
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import DecodedDecision
from sefia.testing import make_function_info, make_tool_call_request


@pytest.mark.parametrize("content", ["", None, "not json"])
@pytest.mark.parametrize(
    "error_type", [DecisionDecodingError, LLMCompletionDecodingError]
)
async def test_repairs_decoding_error_with_rejected_completion(
    transport: AsyncMock,
    make_strategy: Callable[..., LLMInferenceStrategy],
    content: str | None,
    error_type: type[DecisionDecodingError] | type[LLMCompletionDecodingError],
) -> None:
    error = error_type(LLMCompletion(content=content), "response could not be decoded")
    transport.request_decision.side_effect = [
        error,
        transport.request_decision.return_value,
    ]
    publisher = AsyncMock(spec=EventPublisher)

    result = await make_strategy().decide_next_step(
        make_function_info(return_type=str), [], ToolRegistry(), publisher
    )

    assert isinstance(result, ResultDecision) and result.result == "done"
    assert transport.request_decision.await_count == 2
    rejected = transport.request_decision.await_args.kwargs["request"].rejected
    assert rejected.completion is error.completion
    assert rejected.completion.content == content
    assert rejected.reason.endswith("response could not be decoded")
    repairs = [
        c.args[0]
        for c in publisher.publish.await_args_list
        if isinstance(c.args[0], DecisionRepairAttempt)
    ]
    assert len(repairs) == 1
    assert repairs[0].attempt == 1
    assert repairs[0].error.__cause__ is error


@pytest.mark.parametrize(
    "invalid",
    [
        {"decision": "result", "result": None},
        {
            "decision": "tool_calls",
            "tool_calls": [{"name": "unknown", "arguments": dict[str, object]()}],
        },
    ],
)
async def test_repairs_validation_failure_and_forwards_rejected_data(
    transport: AsyncMock,
    make_strategy: Callable[..., LLMInferenceStrategy],
    invalid: dict[str, object],
) -> None:
    content = json.dumps(invalid)
    transport.request_decision.side_effect = [
        DecodedDecision(
            StructuredData.parse_json(content), LLMCompletion(content=content)
        ),
        transport.request_decision.return_value,
    ]
    registry = ToolRegistry()
    registry.add(lambda: None, name="known")

    result = await make_strategy().decide_next_step(
        make_function_info(return_type=str),
        [],
        registry,
        AsyncMock(spec=EventPublisher),
    )

    assert isinstance(result, ResultDecision) and result.result == "done"
    assert transport.request_decision.await_count == 2
    rejected = transport.request_decision.await_args.kwargs["request"].rejected
    assert rejected.completion.content == content
    assert rejected.reason


async def test_repair_preserves_rejected_completion_semantics(
    transport: AsyncMock, make_strategy: Callable[..., LLMInferenceStrategy]
) -> None:
    completion = LLMCompletion(
        tool_calls=[
            ToolCall(
                id="call-1",
                name="unknown",
                arguments=StructuredData.from_json({"query": "lost"}),
            )
        ]
    )
    transport.request_decision.side_effect = [
        DecisionDecodingError(completion, "invalid decision"),
        transport.request_decision.return_value,
    ]

    await make_strategy().decide_next_step(
        make_function_info(return_type=str),
        [],
        ToolRegistry(),
        AsyncMock(spec=EventPublisher),
    )

    rejected = transport.request_decision.await_args.kwargs["request"].rejected
    assert rejected.completion is completion
    assert rejected.reason.endswith("invalid decision")


async def test_repair_preserves_executor_history(
    transport: AsyncMock, make_strategy: Callable[..., LLMInferenceStrategy]
) -> None:
    history = [
        ToolCallsDecision(
            [make_tool_call_request(id="1", name="search", arguments={"q": "x"})]
        ),
        ToolCallResult(tool_call_id="1", result="found"),
    ]
    snapshot = list(history)
    transport.request_decision.side_effect = [
        DecisionDecodingError(LLMCompletion(content="not json"), "invalid"),
        transport.request_decision.return_value,
    ]

    await make_strategy().decide_next_step(
        make_function_info(return_type=str),
        history,
        ToolRegistry(),
        AsyncMock(spec=EventPublisher),
    )

    first, retry = [
        c.kwargs["request"] for c in transport.request_decision.await_args_list
    ]
    assert history == snapshot
    assert first.history is retry.history
    assert first.inference_prompt is retry.inference_prompt
    assert first.rejected is None
    assert retry.rejected is not None


@pytest.mark.parametrize("budget", [0, 2])
async def test_exhausted_budget_preserves_error_and_limits_attempts(
    transport: AsyncMock,
    make_strategy: Callable[..., LLMInferenceStrategy],
    budget: int,
) -> None:
    error = DecisionDecodingError(LLMCompletion(content="not json"), "invalid")
    transport.request_decision.side_effect = error
    publisher = AsyncMock(spec=EventPublisher)

    with pytest.raises(InvalidInferenceResponseError) as exc_info:
        await make_strategy(max_repair_attempts=budget).decide_next_step(
            make_function_info(return_type=str),
            [],
            ToolRegistry(),
            publisher,
        )

    assert transport.request_decision.await_count == budget + 1
    assert exc_info.value.__cause__ is error
    assert exc_info.value.raw_content == "not json"
    repairs = [
        c.args[0]
        for c in publisher.publish.await_args_list
        if isinstance(c.args[0], DecisionRepairAttempt)
    ]
    assert [e.attempt for e in repairs] == list(range(1, budget + 1))


def test_rejects_negative_budget(
    make_strategy: Callable[..., LLMInferenceStrategy],
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_strategy(max_repair_attempts=-1)
