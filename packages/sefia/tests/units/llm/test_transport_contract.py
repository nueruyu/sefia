import inspect
from dataclasses import dataclass
from typing import TypeAlias
from unittest.mock import AsyncMock, Mock

import pytest

import sefia.llm.transports as transports
from sefia.inference import FunctionInfo
from sefia.llm import LLMCompletion, PromptRenderer, ToolCall
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import (
    DecisionRequest,
    DecisionTransport,
    NativeDecisionTransport,
    PromptedDecisionTransport,
    StructuredDecisionTransport,
)
from sefia.pydantic import PydanticModelBackend
from sefia.testing import RecordingDecisionObserver


TransportType: TypeAlias = (
    type[NativeDecisionTransport]
    | type[PromptedDecisionTransport]
    | type[StructuredDecisionTransport]
)
TRANSPORT_TYPES: tuple[TransportType, ...] = (
    NativeDecisionTransport,
    PromptedDecisionTransport,
    StructuredDecisionTransport,
)


@dataclass(frozen=True)
class _TransportCase:
    transport: DecisionTransport
    completion: LLMCompletion


def _completion_for(transport_type: TransportType) -> LLMCompletion:
    if transport_type is NativeDecisionTransport:
        return LLMCompletion(
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="return_result",
                    arguments=StructuredData.from_json({"result": "done"}),
                )
            ]
        )
    if transport_type is PromptedDecisionTransport:
        return LLMCompletion(content='{"decision":"result","result":"done"}')
    return LLMCompletion(
        structured_output=StructuredData.from_json(
            {"decision": "result", "result": "done"}
        )
    )


@pytest.fixture(
    params=TRANSPORT_TYPES, ids=lambda transport_type: transport_type.__name__
)
def transport_case(request: pytest.FixtureRequest) -> _TransportCase:
    transport_type = request.param
    return _TransportCase(transport_type(), _completion_for(transport_type))


def _request() -> DecisionRequest:
    return DecisionRequest(
        function=FunctionInfo(
            qualname="answer",
            instructions="Return the answer.",
            bound_arguments={},
            type_hints={},
            return_type=str,
            args=(),
            kwargs={},
        ),
        decision_spec=DecisionSpec.for_inference(
            output_type=str,
            tools=[],
            result_format_factory=PydanticModelBackend(),
        ),
        history=(),
    )


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in transports.__all__
        if inspect.isclass(value := getattr(transports, name))
        and value is not DecisionTransport
        and issubclass(value, DecisionTransport)
    }

    assert set(TRANSPORT_TYPES) == exported


async def test_contract_returns_decoded_data_with_the_source_completion(
    transport_case: _TransportCase,
) -> None:
    client = AsyncMock()
    client.complete.return_value = transport_case.completion
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "prompt"
    renderer.render_tool_result.return_value = "result"
    observer = RecordingDecisionObserver()

    decoded = await transport_case.transport.request_decision(
        client, renderer, _request(), observer, stream=False
    )

    assert decoded.decision_data.tree == {"decision": "result", "result": "done"}
    assert decoded.completion is transport_case.completion
    assert len(observer.prompts) == 1
    client.complete.assert_awaited_once()
