"""Apply the public decision-transport contract to every built-in transport."""

import json
from collections.abc import Callable
from inspect import signature
from typing import Any, Literal, cast

import pytest
import sefia.llm.transports as transports
from sefia import ToolRegistry
from sefia.llm import JsonSnapshot, LLMCompletion, ToolCall
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.streaming import OutputStreamEvent, StringDelta, StringEnd
from sefia.llm.transports import (
    DecisionTransport,
    NativeDecisionTransport,
    PromptedDecisionTransport,
    StructuredDecisionTransport,
)
from sefia.pydantic import PydanticResultFormatFactory
from sefia.testing import (
    DecisionTransportCase,
    DecisionTransportContract,
    make_decision_request,
)
from typing_extensions import override

DecisionKind = Literal["result", "tool_calls"]


def _case(
    transport: DecisionTransport,
    kind: DecisionKind,
    *,
    native: bool = False,
    structured: bool = False,
) -> DecisionTransportCase:
    def lookup(key: str) -> str:
        return key

    tools = ToolRegistry()
    tools.add(lookup, name="lookup")
    request = make_decision_request(
        DecisionSpec.for_inference(
            output_type=str,
            tools=tools.get_all(),
            result_format_factory=PydanticResultFormatFactory(),
        )
    )
    data: dict[str, Any]
    if kind == "result":
        data = {"decision": "result", "result": "done"}
        call = ToolCall(
            id="call-1",
            name="return_result",
            arguments=JsonSnapshot.capture({"result": "done"}),
        )
        logical_events: tuple[OutputStreamEvent, ...] = (
            StringDelta(("decision",), "result"),
            StringEnd(("decision",), "result"),
            StringDelta(("result",), "done"),
            StringEnd(("result",), "done"),
        )
        native_events: tuple[OutputStreamEvent, ...] = (
            StringDelta(("tool_calls", 0, "name"), "return_result"),
            StringEnd(("tool_calls", 0, "name"), "return_result"),
            StringDelta(("tool_calls", 0, "arguments", "result"), "done"),
            StringEnd(("tool_calls", 0, "arguments", "result"), "done"),
        )
    else:
        data = {
            "decision": "tool_calls",
            "tool_calls": [
                {"name": "lookup", "arguments": {"key": "item"}},
            ],
        }
        call = ToolCall(
            id="call-1",
            name="lookup",
            arguments=JsonSnapshot.capture({"key": "item"}),
        )
        native_events = (
            StringDelta(("tool_calls", 0, "name"), "lookup"),
            StringEnd(("tool_calls", 0, "name"), "lookup"),
            StringDelta(("tool_calls", 0, "arguments", "key"), "item"),
            StringEnd(("tool_calls", 0, "arguments", "key"), "item"),
        )
        logical_events = (
            StringDelta(("decision",), "tool_calls"),
            StringEnd(("decision",), "tool_calls"),
            *native_events,
        )
    expected = JsonSnapshot.capture(data)
    content = json.dumps(data)
    events = native_events if native else logical_events
    return DecisionTransportCase(
        transport=transport,
        completion=(
            LLMCompletion(tool_calls=[call])
            if native
            else LLMCompletion(
                content=content,
                structured_output=expected if structured else None,
            )
        ),
        expected_data=expected,
        request=request,
        content_chunks=("calling a tool",) if native else (content,),
        reasoning_chunks=("reasoning",),
        client_output_events=events if native or structured else (),
        expected_output_events=events,
    )


CASE_FACTORIES: dict[
    type[DecisionTransport], Callable[[DecisionKind], DecisionTransportCase]
] = {
    NativeDecisionTransport: lambda kind: _case(
        NativeDecisionTransport(), kind, native=True
    ),
    PromptedDecisionTransport: lambda kind: _case(PromptedDecisionTransport(), kind),
    StructuredDecisionTransport: lambda kind: _case(
        StructuredDecisionTransport(), kind, structured=True
    ),
}


class TestDecisionTransportContract(DecisionTransportContract):
    _case: DecisionTransportCase

    @pytest.fixture(
        autouse=True,
        params=tuple(CASE_FACTORIES),
        ids=[cls.__name__ for cls in CASE_FACTORIES],
    )
    def _prepare_case(
        self, request: pytest.FixtureRequest, decision_kind: DecisionKind
    ) -> None:
        implementation = cast(type[DecisionTransport], request.param)
        self._case = CASE_FACTORIES[implementation](decision_kind)

    @pytest.fixture(params=("result", "tool_calls"))
    def decision_kind(self, request: pytest.FixtureRequest) -> DecisionKind:
        return cast(DecisionKind, request.param)

    @override
    def make_decision_transport_case(self) -> DecisionTransportCase:
        return self._case


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in transports.__all__
        if isinstance(value := getattr(transports, name), type)
        and value is not DecisionTransport
        and issubclass(value, DecisionTransport)
    }

    assert set(CASE_FACTORIES) == exported


def test_transport_contract_has_no_value_conversion_capability() -> None:
    assert tuple(signature(DecisionTransport.request_decision).parameters) == (
        "self",
        "client",
        "prompt_renderer",
        "request",
        "observer",
        "stream",
    )
