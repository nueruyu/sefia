from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from sefia.inference import FunctionInfo
from sefia.llm import DecisionPrompt, LLMCompletion, Message, PromptRenderer
from sefia.llm.exceptions import DecisionDecodingError
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.streaming import StringEnd
from sefia.llm.transports import DecisionRequest, PromptedDecisionTransport
from sefia.pydantic import PydanticModelBackend
from sefia.testing import RecordingDecisionObserver


def _request() -> DecisionRequest:
    decision = DecisionSpec.for_inference(
        output_type=str,
        tools=[],
        result_format_factory=PydanticModelBackend(),
    )
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


def _renderer(prompt: str = "complete prompt") -> Mock:
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = prompt
    return renderer


async def test_uses_the_rendered_prompt_without_a_model() -> None:
    client = AsyncMock()
    completion = LLMCompletion(content='{"decision":"result","result":"done"}')
    client.complete.return_value = completion
    observer = RecordingDecisionObserver()
    renderer = _renderer()

    decoded = await PromptedDecisionTransport().request_decision(
        client, renderer, _request(), observer, stream=False
    )

    sent = client.complete.await_args.kwargs
    assert sent["messages"] == [Message(role="user", content="complete prompt")]
    assert sent["decision_spec"] is None
    assert observer.prompt == "complete prompt"
    assert decoded.decision_data.tree == {"decision": "result", "result": "done"}
    assert decoded.completion is completion
    rendered_prompt = cast(DecisionPrompt, renderer.render.call_args.args[0])
    assert '{"decision":"result"' in rendered_prompt.response_instructions
    assert "provided structured output schema" not in (
        rendered_prompt.response_instructions
    )


async def test_streams_fenced_json_after_prose() -> None:
    content = (
        "Explanation with {irrelevant} braces.\n"
        "```json\n"
        '{"decision":"tool_calls","tool_calls":'
        '[{"name":"search","arguments":{"query":"sefia"}}]}\n'
        "```"
    )
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(content=content)
    observer = RecordingDecisionObserver()

    await PromptedDecisionTransport().request_decision(
        client, _renderer(), _request(), observer, stream=True
    )
    callback = client.complete.await_args.kwargs["stream_callback"]
    for character in content:
        await callback(character)

    assert StringEnd(("tool_calls", 0, "name"), "search") in observer.output_events


async def test_reports_undecodable_response() -> None:
    client = AsyncMock()
    completion = LLMCompletion(content="not json")
    client.complete.return_value = completion

    with pytest.raises(DecisionDecodingError) as exc_info:
        await PromptedDecisionTransport().request_decision(
            client,
            _renderer(),
            _request(),
            RecordingDecisionObserver(),
            stream=False,
        )

    assert exc_info.value.completion is completion


async def test_reports_text_and_reasoning_progress() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMCompletion(
        content='{"decision":"result","result":"done"}'
    )
    observer = RecordingDecisionObserver()

    await PromptedDecisionTransport().request_decision(
        client, _renderer(), _request(), observer, stream=True
    )

    await client.complete.await_args.kwargs["stream_callback"]("text")
    await client.complete.await_args.kwargs["reasoning_callback"]("reasoning")
    assert observer.response_texts == ["text"]
    assert observer.reasoning_texts == ["reasoning"]
