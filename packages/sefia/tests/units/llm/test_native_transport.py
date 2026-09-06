import json
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from sefia._tool_system import ToolRegistry
from sefia.inference import (
    ToolCallResult,
)
from sefia.llm import (
    DecisionPrompt,
    LLMCompletion,
    PromptRenderer,
    ToolCall,
)
from sefia.llm.exceptions import DecisionDecodingError
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import DecisionRequest, NativeDecisionTransport
from sefia.pydantic import PydanticModelBackend
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
    backend = PydanticModelBackend()
    registry = ToolRegistry()
    for function in functions:
        registry.add(function, name=backend.tool_name(function))
    return DecisionSpec.for_inference(
        output_type=output_type,
        tools=registry.get_all(),
        result_format_factory=backend,
    )


def _request(decision: DecisionSpec) -> DecisionRequest:
    return make_decision_request(decision)


def _renderer() -> Mock:
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "prompt"

    def render_tool_result(result: ToolCallResult) -> str:
        return json.dumps(result.result)

    renderer.render_tool_result.side_effect = render_tool_result
    return renderer


def _call(name: str, arguments: str = "{}") -> ToolCall:
    return ToolCall(
        id="provider-id",
        name=name,
        arguments=StructuredData.parse_json(arguments),
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
        client, renderer, _request(decision), observer, stream=False
    )

    assert decoded.decision_data.tree == {
        "decision": "tool_calls",
        "tool_calls": [{"name": "lookup", "arguments": {"key": "item"}}],
    }
    sent = client.complete.await_args.kwargs
    assert [tool.name for tool in sent["tools"]] == [
        "lookup",
        "return_result",
    ]
    assert sent["decision_spec"] is None
    assert observer.prompt == "prompt"
    rendered_prompt = cast(DecisionPrompt, renderer.render.call_args.args[0])
    assert "return_result" in rendered_prompt.response_instructions
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

    assert decoded.decision_data.tree == {
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
