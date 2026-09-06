from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

import pytest
from sefia import ToolRegistry
from sefia.event_system import EventPublisher
from sefia.inference import ResultDecision
from sefia.llm import (
    LLMClient,
    LLMCompletion,
    LLMInferenceStrategy,
    PromptRenderer,
    ToolCall,
)
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
