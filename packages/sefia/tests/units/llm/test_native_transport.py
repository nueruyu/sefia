from dataclasses import dataclass
from typing import Any, Never, cast
from unittest.mock import AsyncMock, Mock

import pytest
from sefia._tool_system import ToolRegistry
from sefia.llm import (
    InferencePrompt,
    JsonSnapshot,
    LLMCompletion,
    PromptRenderer,
    ToolCall,
)
from sefia.llm.exceptions import DecisionDecodingError
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.transports import (
    DecisionRequest,
    DecisionToolCalls,
    DecisionToolResult,
    NativeDecisionTransport,
)
from sefia.pydantic import (
    PydanticResultFormatFactory,
    PydanticToolFunctionInspector,
)
from sefia.testing import (
    RecordingDecisionObserver,
    make_decision_request,
)


def lookup(key: str) -> str:
    """Look up a value by key."""
    raise NotImplementedError


@dataclass
class Result:
    value: str


def _decision(output_type: Any, *functions: Any) -> DecisionSpec:
    inspector = PydanticToolFunctionInspector()
    registry = ToolRegistry()
    for function in functions:
        registry.add(
            function,
            name=inspector.tool_name(function),
            inspector=inspector,
        )
    return DecisionSpec.for_inference(
        output_type=output_type,
        tools=registry.get_all(),
        result_format_factory=PydanticResultFormatFactory(),
    )


def _request(decision: DecisionSpec) -> DecisionRequest:
    return make_decision_request(decision)


def _renderer() -> Mock:
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "prompt"
    return renderer


def _call(name: str, arguments: str = "{}") -> ToolCall:
    return ToolCall(
        id="provider-id",
        name=name,
        arguments=JsonSnapshot.parse_json(arguments),
    )


async def test_native_transport_exposes_application_and_result_tools() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        tool_calls=[_call("lookup", '{"key":"item"}')]
    )
    decision = _decision(Result, lookup)
    renderer = _renderer()
    observer = RecordingDecisionObserver()

    decoded = await NativeDecisionTransport().request_decision(
        client,
        renderer,
        _request(decision),
        observer,
        stream=False,
    )

    assert decoded.decision_data.to_json_compatible() == {
        "decision": "tool_calls",
        "tool_calls": [{"name": "lookup", "arguments": {"key": "item"}}],
    }
    sent = client.complete.await_args.kwargs
    assert [tool.name for tool in sent["tools"]] == [
        "lookup",
        "return_result",
    ]
    assert sent["decision_spec"] is None
    assert observer.messages == tuple(sent["messages"])
    rendered_prompt = cast(InferencePrompt, renderer.render.call_args.args[0])
    assert "return_result" in sent["messages"][-1].content
    assert rendered_prompt.tools == ()


async def test_native_transport_decodes_typed_result() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        tool_calls=[_call("return_result", '{"result":{"value":"done"}}')]
    )
    decision = _decision(Result)

    decoded = await NativeDecisionTransport().request_decision(
        client,
        _renderer(),
        _request(decision),
        RecordingDecisionObserver(),
        stream=False,
    )

    assert decoded.decision_data.to_json_compatible() == {
        "decision": "result",
        "result": {"value": "done"},
    }


async def test_native_transport_requires_a_tool_call() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(content="done")
    decision = _decision(str)

    with pytest.raises(DecisionDecodingError, match="did not call"):
        await NativeDecisionTransport().request_decision(
            client,
            _renderer(),
            _request(decision),
            RecordingDecisionObserver(),
            stream=False,
        )


async def test_native_transport_forwards_history_in_tool_only_mode() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        tool_calls=[_call("lookup", '{"key":"next"}')]
    )
    request = make_decision_request(
        _decision(Never, lookup),
        history=(
            DecisionToolCalls(
                (
                    ToolCall(
                        id="call-1",
                        name="lookup",
                        arguments=JsonSnapshot.capture({"key": "first"}),
                    ),
                )
            ),
            DecisionToolResult(
                tool_call_id="call-1",
                result=JsonSnapshot.from_scalar("found"),
            ),
        ),
    )

    decoded = await NativeDecisionTransport().request_decision(
        client,
        _renderer(),
        request,
        RecordingDecisionObserver(),
        stream=False,
    )

    sent = client.complete.await_args.kwargs
    assert [tool.name for tool in sent["tools"]] == ["lookup"]
    assert [message.role for message in sent["messages"]] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert sent["messages"][-1].content.startswith("## Response\n\n")
    assert "Call one or more available tools." in sent["messages"][-1].content
    assert sent["messages"][1].tool_calls[0].id == "call-1"
    assert sent["messages"][2].tool_call_id == "call-1"
    assert decoded.decision_data.to_json_compatible() == {
        "decision": "tool_calls",
        "tool_calls": [{"name": "lookup", "arguments": {"key": "next"}}],
    }


async def test_native_transport_uses_collision_free_name_for_prompt_and_decoding() -> (
    None
):
    def return_result(value: str) -> str:
        return value

    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        tool_calls=[_call("return_result_2", '{"result":"done"}')]
    )
    renderer = _renderer()
    registry = ToolRegistry()
    registry.add(return_result, name="return_result")
    decision = DecisionSpec.for_inference(
        output_type=str,
        tools=registry.get_all(),
        result_format_factory=PydanticResultFormatFactory(),
    )
    decoded = await NativeDecisionTransport().request_decision(
        client,
        renderer,
        _request(decision),
        RecordingDecisionObserver(),
        stream=False,
    )

    sent = client.complete.await_args.kwargs
    assert [tool.name for tool in sent["tools"]] == ["return_result", "return_result_2"]
    assert "return_result_2" in sent["messages"][-1].content
    assert decoded.decision_data.to_json_compatible() == {
        "decision": "result",
        "result": "done",
    }
