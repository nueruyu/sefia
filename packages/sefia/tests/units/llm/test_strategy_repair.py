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
from sefia.llm.json import JsonSnapshot
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
            JsonSnapshot.parse_json(content), LLMCompletion(content=content)
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
                arguments=JsonSnapshot.capture({"query": "lost"}),
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
    assert first.function is retry.function
    assert first.arguments is retry.arguments
    assert first.rejected is None
    assert retry.rejected is not None


async def test_generic_inference_failure_is_not_converted_to_repair_feedback(
    transport: AsyncMock, make_strategy: Callable[..., LLMInferenceStrategy]
) -> None:
    error = InvalidInferenceResponseError("recover on a new inference attempt")
    transport.request_decision.side_effect = error
    publisher = AsyncMock(spec=EventPublisher)

    with pytest.raises(InvalidInferenceResponseError) as exc_info:
        await make_strategy().decide_next_step(
            make_function_info(return_type=str),
            [],
            ToolRegistry(),
            publisher,
        )

    assert exc_info.value is error
    transport.request_decision.assert_awaited_once()
    assert not any(
        isinstance(call.args[0], DecisionRepairAttempt)
        for call in publisher.publish.await_args_list
    )


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


async def test_repair_reuses_materialized_inputs(
    transport: AsyncMock, make_strategy: Callable[..., LLMInferenceStrategy]
) -> None:
    from sefia.llm import JsonCompatible, JsonMaterializer
    from sefia.llm.transports import DecisionRequest, DecisionToolResult
    from sefia.pydantic import PydanticJsonMaterializer
    from typing_extensions import override

    class CountingMaterializer(JsonMaterializer):
        def __init__(self) -> None:
            self.calls = 0

        @override
        def materialize(self, value: object) -> JsonCompatible:
            self.calls += 1
            return PydanticJsonMaterializer().materialize(value)

    materializer = CountingMaterializer()
    strategy = make_strategy()
    strategy._json_materializer = materializer
    argument = [1]
    result = {"items": [2]}
    requests: list[DecisionRequest] = []
    valid = transport.request_decision.return_value

    async def complete(**kwargs: object) -> DecodedDecision:
        request = kwargs["request"]
        assert isinstance(request, DecisionRequest)
        requests.append(request)
        assert materializer.calls == 2
        assert request.arguments.to_json_compatible() == {"value": [1]}
        item = request.history[0]
        assert isinstance(item, DecisionToolResult)
        assert item.result.to_json_compatible() == {"items": [2]}
        if len(requests) == 1:
            argument.append(9)
            result["items"].append(9)
            raise DecisionDecodingError(LLMCompletion(content="invalid"), "invalid")
        return valid

    transport.request_decision.side_effect = complete
    await strategy.decide_next_step(
        make_function_info(return_type=str, bound_arguments={"value": argument}),
        [ToolCallResult(tool_call_id="1", result=result)],
        ToolRegistry(),
        AsyncMock(spec=EventPublisher),
    )
    first, repaired = requests
    assert first.arguments is repaired.arguments
    assert first.history is repaired.history
    assert first.messages_before is repaired.messages_before
    assert first.messages_after is repaired.messages_after
