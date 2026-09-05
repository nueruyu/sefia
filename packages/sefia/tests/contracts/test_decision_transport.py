"""Apply the public decision-transport contract to every built-in transport."""

import inspect
from typing import TypeAlias

import pytest

import sefia.llm.transports as transports
from sefia.llm import LLMCompletion, ToolCall
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import (
    DecisionTransport,
    NativeDecisionTransport,
    PromptedDecisionTransport,
    StructuredDecisionTransport,
)
from sefia.testing import DecisionTransportCase, DecisionTransportContract

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

_EXPECTED = StructuredData.from_json({"decision": "result", "result": "done"})


def _case(transport_type: TransportType) -> DecisionTransportCase:
    if transport_type is NativeDecisionTransport:
        completion = LLMCompletion(
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="return_result",
                    arguments=StructuredData.from_json({"result": "done"}),
                )
            ]
        )
    elif transport_type is PromptedDecisionTransport:
        completion = LLMCompletion(content='{"decision":"result","result":"done"}')
    else:
        completion = LLMCompletion(structured_output=_EXPECTED)
    return DecisionTransportCase(transport_type(), completion, _EXPECTED)


class TestNativeDecisionTransportContract(DecisionTransportContract):
    @pytest.fixture
    def decision_transport_case(self) -> DecisionTransportCase:
        return _case(NativeDecisionTransport)


class TestPromptedDecisionTransportContract(DecisionTransportContract):
    @pytest.fixture
    def decision_transport_case(self) -> DecisionTransportCase:
        return _case(PromptedDecisionTransport)


class TestStructuredDecisionTransportContract(DecisionTransportContract):
    @pytest.fixture
    def decision_transport_case(self) -> DecisionTransportCase:
        return _case(StructuredDecisionTransport)


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in transports.__all__
        if inspect.isclass(value := getattr(transports, name))
        and value is not DecisionTransport
        and issubclass(value, DecisionTransport)
    }

    assert set(TRANSPORT_TYPES) == exported
