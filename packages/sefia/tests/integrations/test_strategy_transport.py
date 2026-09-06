from dataclasses import dataclass
from typing import Never
from unittest.mock import AsyncMock, Mock

import pytest
from sefia import ToolRegistry
from sefia.event_system import EventPublisher
from sefia.exceptions import InvalidInferenceResponseError
from sefia.inference import ResultDecision, ToolCallsDecision
from sefia.llm import (
    LLMClient,
    LLMCompletion,
    LLMInferenceStrategy,
    PromptRenderer,
    ToolCall,
)
from sefia.llm.step_decision import StepDecisionMode
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import (
    DecisionTransport,
    NativeDecisionTransport,
    StructuredDecisionTransport,
)
from sefia.pydantic import PydanticModelBackend
from sefia.testing import make_function_info


@dataclass
class Result:
    value: str


@pytest.mark.parametrize(
    ("transport", "valid"),
    [
        (
            StructuredDecisionTransport(),
            LLMCompletion(
                structured_output=StructuredData.from_json(
                    {"decision": "result", "result": {"value": "done"}}
                )
            ),
        ),
        (
            NativeDecisionTransport(),
            LLMCompletion(
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="return_result",
                        arguments=StructuredData.from_json(
                            {"result": {"value": "done"}}
                        ),
                    )
                ]
            ),
        ),
    ],
    ids=["structured", "native"],
)
async def test_transport_feedback_reaches_renderer_and_result_is_restored(
    transport: DecisionTransport, valid: LLMCompletion
) -> None:
    client = AsyncMock(spec=LLMClient)
    client.complete.side_effect = [LLMCompletion(content="invalid"), valid]
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "prompt"
    strategy = LLMInferenceStrategy(client, PydanticModelBackend(), renderer, transport)

    decision = await strategy.decide_next_step(
        make_function_info(return_type=Result),
        [],
        ToolRegistry(),
        AsyncMock(spec=EventPublisher),
    )

    assert isinstance(decision, ResultDecision)
    assert decision.result == Result("done")
    first, retry = [c.args[0] for c in renderer.render.call_args_list]
    assert first.rejected is None
    assert retry.rejected.content == "invalid"
    assert retry.rejected.reason
    assert client.complete.await_count == 2
    sent = client.complete.await_args.kwargs
    assert sent["stream_callback"] is None
    assert sent["reasoning_callback"] is None


@pytest.mark.parametrize(
    "transport", [StructuredDecisionTransport(), NativeDecisionTransport()]
)
@pytest.mark.parametrize(
    "returns_result", [False, True], ids=["tool-call", "forbidden-result"]
)
async def test_never_mode_is_preserved_through_strategy_and_transport(
    transport: DecisionTransport, returns_result: bool
) -> None:
    tools = ToolRegistry()
    tools.add(lambda: "ok", name="lookup")
    data = StructuredData.from_json(
        {"decision": "result", "result": "done"}
        if returns_result
        else {
            "decision": "tool_calls",
            "tool_calls": [{"name": "lookup", "arguments": {}}],
        }
    )
    completion = LLMCompletion(
        structured_output=data,
        tool_calls=[
            ToolCall(
                id="provider-id",
                name="return_result" if returns_result else "lookup",
                arguments=StructuredData.from_json(
                    {"result": "done"} if returns_result else {}
                ),
            )
        ],
    )
    client = AsyncMock(spec=LLMClient)
    client.complete.return_value = completion
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "prompt"
    strategy = LLMInferenceStrategy(
        client, PydanticModelBackend(), renderer, transport, max_repair_attempts=0
    )
    function = make_function_info(return_type=Never)
    publisher = AsyncMock(spec=EventPublisher)

    if returns_result:
        with pytest.raises(InvalidInferenceResponseError):
            await strategy.decide_next_step(function, [], tools, publisher)
    else:
        decision = await strategy.decide_next_step(function, [], tools, publisher)
        assert isinstance(decision, ToolCallsDecision)
        assert [call.name for call in decision.calls] == ["lookup"]

    prompt = renderer.render.call_args.args[0]
    before = publisher.publish.await_args_list[0].args[0]
    assert before.decision_spec.mode is StepDecisionMode.TOOLS_REQUIRED
    assert "Call one or more available tools." in prompt.response_instructions
