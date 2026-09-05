from collections.abc import Awaitable, Callable
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from sefia.llm import DecisionPrompt, LLMCompletion, Message, PromptRenderer
from sefia.llm.exceptions import DecisionDecodingError
from sefia.llm.structured_data import StructuredData
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.streaming import StringEnd
from sefia.llm.transports import DecisionRequest, StructuredDecisionTransport
from sefia.pydantic import PydanticModelBackend
from sefia.testing import RecordingDecisionObserver, make_decision_request


def _request() -> DecisionRequest:
    decision = DecisionSpec.for_inference(
        output_type=str,
        tools=[],
        result_format_factory=PydanticModelBackend(),
    )
    return make_decision_request(decision)


def _renderer(prompt: str = "complete prompt") -> Mock:
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = prompt
    return renderer


async def test_renders_and_delivers_one_complete_prompt() -> None:
    client = AsyncMock()
    completion = LLMCompletion(
        structured_output=StructuredData.from_json(
            {"decision": "result", "result": "done"}
        )
    )
    client.complete.return_value = completion
    renderer = _renderer()
    request = _request()
    observer = RecordingDecisionObserver()

    decoded = await StructuredDecisionTransport().request_decision(
        client, renderer, request, observer, stream=False
    )

    sent = client.complete.await_args.kwargs
    assert sent["messages"] == [Message(role="user", content="complete prompt")]
    assert sent["decision_spec"] is request.decision_spec
    assert observer.prompt == "complete prompt"
    assert decoded.decision_data.tree == {"decision": "result", "result": "done"}
    assert decoded.completion is completion
    renderer.render.assert_called_once()
    rendered_prompt = cast(DecisionPrompt, renderer.render.call_args.args[0])
    assert "provided structured output schema" in (
        rendered_prompt.response_instructions
    )
    assert '"decision"' not in rendered_prompt.response_instructions


async def test_observer_finishes_before_the_client_request() -> None:
    order: list[str] = []
    client = AsyncMock()

    async def complete(**_kwargs: object) -> LLMCompletion:
        order.append("request")
        return LLMCompletion(
            structured_output=StructuredData.from_json(
                {"decision": "result", "result": "done"}
            )
        )

    client.complete.side_effect = complete

    class Observer(RecordingDecisionObserver):
        async def before_request(self, prompt: str) -> None:
            await super().before_request(prompt)
            order.append("observed")

    await StructuredDecisionTransport().request_decision(
        client, _renderer(), _request(), Observer(), stream=False
    )

    assert order == ["observed", "request"]


async def test_reports_undecodable_response() -> None:
    client = AsyncMock()
    completion = LLMCompletion(content="not json")
    client.complete.return_value = completion

    with pytest.raises(DecisionDecodingError) as exc_info:
        await StructuredDecisionTransport().request_decision(
            client,
            _renderer(),
            _request(),
            RecordingDecisionObserver(),
            stream=False,
        )

    assert exc_info.value.completion is completion


async def test_rejects_raw_json_content() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        content='{"decision":"result","result":"done"}'
    )

    with pytest.raises(DecisionDecodingError, match="structured output"):
        await StructuredDecisionTransport().request_decision(
            client,
            _renderer(),
            _request(),
            RecordingDecisionObserver(),
            stream=False,
        )


async def test_reports_text_and_reasoning_progress() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        structured_output=StructuredData.from_json(
            {"decision": "result", "result": "done"}
        )
    )
    observer = RecordingDecisionObserver()

    await StructuredDecisionTransport().request_decision(
        client, _renderer(), _request(), observer, stream=True
    )

    await client.complete.await_args.kwargs["stream_callback"]("text")
    await client.complete.await_args.kwargs["reasoning_callback"]("reasoning")
    assert observer.response_texts == ["text"]
    assert observer.reasoning_texts == ["reasoning"]


async def test_reports_logical_tool_progress() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        structured_output=StructuredData.from_json(
            {"decision": "tool_calls", "tool_calls": []}
        )
    )
    observer = RecordingDecisionObserver()

    await StructuredDecisionTransport().request_decision(
        client, _renderer(), _request(), observer, stream=True
    )

    callback = cast(
        Callable[[StringEnd], Awaitable[None]],
        client.complete.await_args.kwargs["output_callback"],
    )
    await callback(StringEnd(("tool_calls", 0, "name"), "search"))
    assert observer.output_events == [StringEnd(("tool_calls", 0, "name"), "search")]
