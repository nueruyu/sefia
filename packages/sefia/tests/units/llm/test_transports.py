from collections.abc import Awaitable, Callable
from typing import cast
from unittest.mock import AsyncMock

import pytest

from sefia.llm import LLMResponse
from sefia.llm.llm_output import LLMOutput
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.streaming import StringEnd
from sefia.llm.transports import (
    DecisionDecodingError,
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


async def test_structured_transport_delivers_one_complete_prompt() -> None:
    client = AsyncMock()
    raw = LLMResponse(
        structured_output=LLMOutput.from_json({"decision": "result", "result": "done"})
    )
    client.complete.return_value = raw
    decision = _decision()

    response = await StructuredDecisionTransport().complete(
        client=client,
        prompt="complete prompt",
        decision=decision,
        progress=None,
    )

    request = client.complete.await_args.kwargs
    assert [message.to_dict(exclude_none=True) for message in request["messages"]] == [
        {"role": "user", "content": "complete prompt"}
    ]
    assert request["decision_model"] is decision
    assert response.output.data == {"decision": "result", "result": "done"}
    assert response.raw is raw


async def test_prompted_transport_delivers_the_same_prompt_and_decision_shape() -> None:
    client = AsyncMock()
    raw = LLMResponse(content='{"decision":"result","result":"done"}')
    client.complete.return_value = raw

    response = await PromptedDecisionTransport().complete(
        client=client,
        prompt="complete prompt",
        decision=_decision(),
        progress=None,
    )

    request = client.complete.await_args.kwargs
    assert [message.to_dict(exclude_none=True) for message in request["messages"]] == [
        {"role": "user", "content": "complete prompt"}
    ]
    assert request["decision_model"] is None
    assert response.output.data == {"decision": "result", "result": "done"}
    assert response.raw is raw


async def test_structured_transport_reports_undecodable_response() -> None:
    client = AsyncMock()
    raw = LLMResponse(content="not json")
    client.complete.return_value = raw

    with pytest.raises(DecisionDecodingError) as exc_info:
        await StructuredDecisionTransport().complete(
            client=client,
            prompt="prompt",
            decision=_decision(),
            progress=None,
        )

    assert exc_info.value.response is raw


async def test_structured_transport_rejects_raw_json_content() -> None:
    client = AsyncMock()
    raw = LLMResponse(content='{"decision":"result","result":"done"}')
    client.complete.return_value = raw

    with pytest.raises(DecisionDecodingError, match="structured output"):
        await StructuredDecisionTransport().complete(
            client=client,
            prompt="prompt",
            decision=_decision(),
            progress=None,
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
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    await PromptedDecisionTransport().complete(client, "prompt", _decision(), collect)
    callback = client.complete.await_args.kwargs["stream_callback"]
    for character in content:
        await callback(character)

    assert ToolCallIdentified(index=0, name="search") in events


async def test_prompted_transport_reports_undecodable_response() -> None:
    client = AsyncMock()
    raw = LLMResponse(content="not json")
    client.complete.return_value = raw

    with pytest.raises(DecisionDecodingError) as exc_info:
        await PromptedDecisionTransport().complete(
            client=client,
            prompt="prompt",
            decision=_decision(),
            progress=None,
        )

    assert exc_info.value.response is raw


@pytest.mark.parametrize(
    ("transport", "response"),
    [
        (
            StructuredDecisionTransport(),
            LLMResponse(
                structured_output=LLMOutput.from_json(
                    {"decision": "result", "result": "done"}
                )
            ),
        ),
        (
            PromptedDecisionTransport(),
            LLMResponse(content='{"decision":"result","result":"done"}'),
        ),
    ],
)
async def test_transports_report_text_and_reasoning_progress(
    transport: DecisionTransport,
    response: LLMResponse,
) -> None:
    client = AsyncMock()
    client.complete.return_value = response
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    await transport.complete(client, "prompt", _decision(), collect)

    await client.complete.await_args.kwargs["stream_callback"]("text")
    await client.complete.await_args.kwargs["reasoning_callback"]("reasoning")

    assert ResponseTextDelta("text") in events
    assert ReasoningTextDelta("reasoning") in events


async def test_structured_transport_reports_logical_tool_progress() -> None:
    client = AsyncMock()
    client.complete.return_value = LLMResponse(
        structured_output=LLMOutput.from_json(
            {"decision": "tool_calls", "tool_calls": []}
        )
    )
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    await StructuredDecisionTransport().complete(
        client=client,
        prompt="prompt",
        decision=_decision(),
        progress=collect,
    )

    callback = cast(
        Callable[[StringEnd], Awaitable[None]],
        client.complete.await_args.kwargs["output_callback"],
    )
    await callback(StringEnd(("tool_calls", 0, "name"), "search"))

    assert events == [ToolCallIdentified(index=0, name="search")]
