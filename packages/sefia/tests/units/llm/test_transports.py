from collections.abc import Awaitable, Callable
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from sefia.inference import FunctionInfo
from sefia.llm import DecisionPrompt, LLMResponse, PromptRenderer
from sefia.llm.llm_output import LLMOutput
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.streaming import StringEnd
from sefia.llm.transports import (
    DecisionDecodingError,
    DecisionProgress,
    DecisionRequest,
    DecisionTransport,
    PromptedDecisionTransport,
    ReasoningTextDelta,
    ResponseTextDelta,
    StructuredDecisionTransport,
    ToolCallIdentified,
)
from sefia.pydantic import PydanticModelBackend


def _decision() -> DecisionSpec:
    return DecisionSpec.for_inference(
        output_type=str,
        tools=[],
        result_format_factory=PydanticModelBackend(),
    )


def _request() -> DecisionRequest:
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
        decision=_decision(),
        history=(),
    )


def _transport(
    transport_type: type[DecisionTransport],
    prompt: str = "complete prompt",
) -> tuple[DecisionTransport, Mock]:
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = prompt
    return transport_type(), renderer


class _RecordingObserver:
    def __init__(self) -> None:
        self.prompt: str | None = None
        self.events: list[DecisionProgress] = []

    async def before_request(self, prompt: str) -> None:
        self.prompt = prompt

    async def progress(self, progress: DecisionProgress) -> None:
        self.events.append(progress)


async def test_structured_transport_renders_and_delivers_one_complete_prompt() -> None:
    client = AsyncMock()
    raw = LLMResponse(
        structured_output=LLMOutput.from_json({"decision": "result", "result": "done"})
    )
    client.complete.return_value = raw
    transport, renderer = _transport(StructuredDecisionTransport)
    request = _request()
    observer = _RecordingObserver()

    response = await transport.request_decision(
        client, renderer, request, observer, stream=False
    )

    sent = client.complete.await_args.kwargs
    assert [message.to_dict(exclude_none=True) for message in sent["messages"]] == [
        {"role": "user", "content": "complete prompt"}
    ]
    assert sent["decision_model"] is request.decision
    assert observer.prompt == "complete prompt"
    assert response.output.data == {"decision": "result", "result": "done"}
    assert response.raw is raw
    renderer.render.assert_called_once()
    rendered_prompt = cast(DecisionPrompt, renderer.render.call_args.args[0])
    assert "Return exactly one JSON object" in rendered_prompt.response_instructions
    assert "JSON Schema" in rendered_prompt.response_instructions


async def test_prompted_transport_uses_the_rendered_prompt_without_a_model() -> None:
    client = AsyncMock()
    raw = LLMResponse(content='{"decision":"result","result":"done"}')
    client.complete.return_value = raw
    transport, renderer = _transport(PromptedDecisionTransport)
    observer = _RecordingObserver()

    response = await transport.request_decision(
        client, renderer, _request(), observer, stream=False
    )

    sent = client.complete.await_args.kwargs
    assert [message.to_dict(exclude_none=True) for message in sent["messages"]] == [
        {"role": "user", "content": "complete prompt"}
    ]
    assert sent["decision_model"] is None
    assert observer.prompt == "complete prompt"
    assert response.output.data == {"decision": "result", "result": "done"}
    assert response.raw is raw


async def test_observer_finishes_before_the_client_request() -> None:
    order: list[str] = []
    client = AsyncMock()

    async def complete(**_kwargs: object) -> LLMResponse:
        order.append("request")
        return LLMResponse(
            structured_output=LLMOutput.from_json(
                {"decision": "result", "result": "done"}
            )
        )

    client.complete.side_effect = complete
    transport, renderer = _transport(StructuredDecisionTransport)

    class Observer(_RecordingObserver):
        async def before_request(self, prompt: str) -> None:
            await super().before_request(prompt)
            order.append("observed")

    await transport.request_decision(
        client, renderer, _request(), Observer(), stream=False
    )

    assert order == ["observed", "request"]


async def test_structured_transport_reports_undecodable_response() -> None:
    client = AsyncMock()
    raw = LLMResponse(content="not json")
    client.complete.return_value = raw
    transport, renderer = _transport(StructuredDecisionTransport)

    with pytest.raises(DecisionDecodingError) as exc_info:
        await transport.request_decision(
            client, renderer, _request(), _RecordingObserver(), stream=False
        )

    assert exc_info.value.response is raw


async def test_structured_transport_rejects_raw_json_content() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMResponse(
        content='{"decision":"result","result":"done"}'
    )
    transport, renderer = _transport(StructuredDecisionTransport)

    with pytest.raises(DecisionDecodingError, match="structured output"):
        await transport.request_decision(
            client, renderer, _request(), _RecordingObserver(), stream=False
        )


async def test_prompted_transport_streams_fenced_json_after_prose() -> None:
    content = (
        "Explanation with {irrelevant} braces.\n"
        "```json\n"
        '{"decision":"tool_calls","tool_calls":'
        '[{"name":"search","arguments":{"query":"sefia"}}]}\n'
        "```"
    )
    client = AsyncMock()
    client.complete.return_value = LLMResponse(content=content)
    transport, renderer = _transport(PromptedDecisionTransport)
    observer = _RecordingObserver()

    await transport.request_decision(
        client, renderer, _request(), observer, stream=True
    )
    callback = client.complete.await_args.kwargs["stream_callback"]
    for character in content:
        await callback(character)

    assert ToolCallIdentified(index=0, name="search") in observer.events


async def test_prompted_transport_reports_undecodable_response() -> None:
    client = AsyncMock()
    raw = LLMResponse(content="not json")
    client.complete.return_value = raw
    transport, renderer = _transport(PromptedDecisionTransport)

    with pytest.raises(DecisionDecodingError) as exc_info:
        await transport.request_decision(
            client, renderer, _request(), _RecordingObserver(), stream=False
        )

    assert exc_info.value.response is raw


@pytest.mark.parametrize(
    ("transport_type", "response"),
    [
        (
            StructuredDecisionTransport,
            LLMResponse(
                structured_output=LLMOutput.from_json(
                    {"decision": "result", "result": "done"}
                )
            ),
        ),
        (
            PromptedDecisionTransport,
            LLMResponse(content='{"decision":"result","result":"done"}'),
        ),
    ],
)
async def test_transports_report_text_and_reasoning_progress(
    transport_type: type[DecisionTransport],
    response: LLMResponse,
) -> None:
    client = AsyncMock()
    client.complete.return_value = response
    transport, renderer = _transport(transport_type)
    observer = _RecordingObserver()

    await transport.request_decision(
        client, renderer, _request(), observer, stream=True
    )

    await client.complete.await_args.kwargs["stream_callback"]("text")
    await client.complete.await_args.kwargs["reasoning_callback"]("reasoning")

    assert ResponseTextDelta("text") in observer.events
    assert ReasoningTextDelta("reasoning") in observer.events


async def test_structured_transport_reports_logical_tool_progress() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMResponse(
        structured_output=LLMOutput.from_json(
            {"decision": "tool_calls", "tool_calls": []}
        )
    )
    transport, renderer = _transport(StructuredDecisionTransport)
    observer = _RecordingObserver()

    await transport.request_decision(
        client, renderer, _request(), observer, stream=True
    )

    callback = cast(
        Callable[[StringEnd], Awaitable[None]],
        client.complete.await_args.kwargs["output_callback"],
    )
    await callback(StringEnd(("tool_calls", 0, "name"), "search"))

    assert observer.events == [ToolCallIdentified(index=0, name="search")]
