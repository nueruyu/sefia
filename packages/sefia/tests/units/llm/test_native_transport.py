import json
from dataclasses import dataclass
from typing import Any, Never, cast
from unittest.mock import AsyncMock, Mock

import pytest

from sefia._tool_system import ToolRegistry
from sefia.inference import (
    FunctionInfo,
    ResultDecision,
    ToolCallRequest,
    ToolCallResult,
    ToolCallsDecision,
)
from sefia.llm import (
    DecisionPrompt,
    LLMCompletion,
    PromptRenderer,
    ToolCall,
)
from sefia.llm._tool_call_ids import ToolCallIdRegistry
from sefia.llm.exceptions import DecisionDecodingError
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.structured_data import StructuredData
from sefia.llm.streaming import OutputStreamEvent, StringEnd as OutputStringEnd
from sefia.llm.transports import (
    DecisionObserver,
    DecisionRequest,
    NativeDecisionTransport,
)
from sefia.pydantic import PydanticModelBackend


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
    return DecisionRequest(
        function=FunctionInfo(
            qualname="test",
            instructions="instructions",
            bound_arguments={},
            type_hints={},
            return_type=str,
            args=(),
            kwargs={},
        ),
        decision_spec=decision,
        history=(),
    )


def _renderer() -> Mock:
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "prompt"

    def render_tool_result(result: ToolCallResult) -> str:
        return json.dumps(result.result)

    renderer.render_tool_result.side_effect = render_tool_result
    return renderer


class _RecordingObserver(DecisionObserver):
    def __init__(self) -> None:
        self.prompt: str | None = None
        self.response_texts: list[str] = []
        self.reasoning_texts: list[str] = []
        self.output_events: list[OutputStreamEvent] = []

    async def before_request(self, prompt: str) -> None:
        self.prompt = prompt

    async def response_text(self, text: str) -> None:
        self.response_texts.append(text)

    async def reasoning_text(self, text: str) -> None:
        self.reasoning_texts.append(text)

    async def output(self, event: OutputStreamEvent) -> None:
        self.output_events.append(event)


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
    observer = _RecordingObserver()

    decoded = await NativeDecisionTransport().request_decision(
        client, renderer, _request(decision), observer, stream=False
    )

    validated = decision.validate(decoded.decision_data, ToolCallIdRegistry())
    assert isinstance(validated, ToolCallsDecision)
    assert validated.calls[0].arguments == {"key": "item"}
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
        _RecordingObserver(),
        stream=False,
    )

    validated = decision.validate(decoded.decision_data, None)
    assert isinstance(validated, ResultDecision)
    assert validated.result == Result("done")


async def test_native_transport_avoids_result_tool_name_collision() -> None:
    def return_result(value: str) -> str:
        return value

    backend = PydanticModelBackend()
    registry = ToolRegistry()
    registry.add(return_result, name="return_result")
    decision = DecisionSpec.for_inference(
        output_type=str,
        tools=registry.get_all(),
        result_format_factory=backend,
    )
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        tool_calls=[_call("return_result_2", '{"result":"done"}')]
    )
    renderer = _renderer()

    await NativeDecisionTransport().request_decision(
        client,
        renderer,
        _request(decision),
        _RecordingObserver(),
        stream=False,
    )

    assert [tool.name for tool in client.complete.await_args.kwargs["tools"]] == [
        "return_result",
        "return_result_2",
    ]
    rendered_prompt = cast(DecisionPrompt, renderer.render.call_args.args[0])
    assert "return_result_2" in rendered_prompt.response_instructions


async def test_native_transport_requires_a_tool_call() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(content="done")
    decision = _decision(str)

    with pytest.raises(DecisionDecodingError, match="did not call"):
        await NativeDecisionTransport().request_decision(
            client,
            _renderer(),
            _request(decision),
            _RecordingObserver(),
            stream=False,
        )


async def test_native_transport_sends_previous_calls_as_native_history() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        tool_calls=[_call("lookup", '{"key":"next"}')]
    )
    decision = _decision(Never, lookup)
    request = _request(decision)
    request = DecisionRequest(
        function=request.function,
        decision_spec=request.decision_spec,
        history=(
            ToolCallsDecision(
                [
                    ToolCallRequest(
                        id="call-1",
                        name="lookup",
                        arguments={"key": "first"},
                    )
                ]
            ),
            ToolCallResult(tool_call_id="call-1", result={"value": "found"}),
        ),
    )
    renderer = _renderer()

    await NativeDecisionTransport().request_decision(
        client,
        renderer,
        request,
        _RecordingObserver(),
        stream=False,
    )

    messages = client.complete.await_args.kwargs["messages"]
    assert [message.role for message in messages] == ["user", "assistant", "tool"]
    assert messages[1].tool_calls == [
        ToolCall(
            id="call-1",
            name="lookup",
            arguments=StructuredData.from_json({"key": "first"}),
        )
    ]
    assert messages[2].tool_call_id == "call-1"
    assert messages[2].content == '{"value": "found"}'
    rendered_prompt = cast(DecisionPrompt, renderer.render.call_args.args[0])
    assert rendered_prompt.history == ()


async def test_native_transport_requires_object_arguments() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        tool_calls=[
            ToolCall(
                id="provider-id",
                name="lookup",
                arguments=StructuredData.from_json([]),
            )
        ]
    )
    decision = _decision(Never, lookup)

    with pytest.raises(DecisionDecodingError):
        await NativeDecisionTransport().request_decision(
            client,
            _renderer(),
            _request(decision),
            _RecordingObserver(),
            stream=False,
        )


async def test_native_transport_forwards_all_progress_kinds() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        tool_calls=[_call("lookup", '{"key":"item"}')]
    )
    observer = _RecordingObserver()

    async def native_complete(**kwargs: Any) -> LLMCompletion:
        await kwargs["stream_callback"]("text")
        await kwargs["reasoning_callback"]("reasoning")
        await kwargs["output_callback"](
            OutputStringEnd(("tool_calls", 0, "name"), "lookup")
        )
        await kwargs["output_callback"](
            OutputStringEnd(("tool_calls", 0, "arguments", "key"), "item")
        )
        return client.complete.return_value

    client.complete.side_effect = native_complete
    decision = _decision(Never, lookup)
    await NativeDecisionTransport().request_decision(
        client,
        _renderer(),
        _request(decision),
        observer,
        stream=True,
    )

    assert observer.response_texts == ["text"]
    assert observer.reasoning_texts == ["reasoning"]
    assert observer.output_events == [
        OutputStringEnd(("tool_calls", 0, "name"), "lookup"),
        OutputStringEnd(("tool_calls", 0, "arguments", "key"), "item"),
    ]
