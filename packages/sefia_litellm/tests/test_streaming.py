from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from litellm import (
    ChatCompletionMessageToolCall,
    Choices,
    Message as LiteLLMMessage,
    ModelResponse,
)
from pytest_mock import MockerFixture
from sefia.llm.exceptions import LLMResponseDecodingError
from sefia.llm.json_schema import JsonSchemaDocument
from sefia.llm.streaming import (
    OutputStreamEvent,
    Scalar as OutputScalar,
    StringEnd as OutputStringEnd,
)
from sefia_litellm._schema import StructuredValueFormat
from sefia_litellm._streaming import (
    _extract_native_tool_call_fragments,
    handle_stream,
)


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_name: str | None = None,
    tool_arguments: str | None = None,
) -> SimpleNamespace:
    tool_calls = None
    if tool_name is not None or tool_arguments is not None:
        tool_calls = [
            SimpleNamespace(
                index=0,
                function=SimpleNamespace(
                    name=tool_name,
                    arguments=tool_arguments,
                ),
            )
        ]
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


async def _stream(*chunks: SimpleNamespace) -> AsyncIterator[SimpleNamespace]:
    for chunk in chunks:
        yield chunk


def _tool_response(name: str, arguments: str) -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=LiteLLMMessage(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call-1",
                            function={"name": name, "arguments": arguments},
                            type="function",
                        )
                    ],
                ),
            )
        ]
    )


def test_extracts_native_tool_call_fragments_without_decoding_json() -> None:
    fragments = _extract_native_tool_call_fragments(
        [
            SimpleNamespace(
                index=2,
                function=SimpleNamespace(name="lookup", arguments='{"key":"'),
            )
        ]
    )

    assert len(fragments) == 1
    assert fragments[0].index == 2
    assert fragments[0].name == "lookup"
    assert fragments[0].arguments_json == '{"key":"'


async def test_routes_reasoning_and_content_separately(
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "litellm.stream_chunk_builder",
        return_value=ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(role="assistant", content="done"),
                )
            ]
        ),
    )
    content_tokens: list[str] = []
    reasoning_tokens: list[str] = []

    async def on_content(token: str) -> None:
        content_tokens.append(token)

    async def on_reasoning(token: str) -> None:
        reasoning_tokens.append(token)

    response = await handle_stream(
        _stream(
            _chunk(reasoning="Let me "),
            _chunk(reasoning="think."),
            _chunk(content="answer"),
        ),
        content_callback=on_content,
        output_callback=None,
        reasoning_callback=on_reasoning,
        messages=[],
        output=None,
        requested_model="gpt-4o",
    )

    assert reasoning_tokens == ["Let me ", "think."]
    assert content_tokens == ["answer"]
    assert response.reasoning_content == "Let me think."


async def test_invalid_built_response_is_a_decoding_error(
    mocker: MockerFixture,
) -> None:
    mocker.patch("litellm.stream_chunk_builder", return_value=None)

    with pytest.raises(
        LLMResponseDecodingError, match="could not reconstruct"
    ) as exc_info:
        await handle_stream(
            _stream(
                _chunk(reasoning="partial thought"),
                _chunk(content="partial answer"),
            ),
            content_callback=None,
            output_callback=None,
            reasoning_callback=None,
            messages=[],
            output=None,
            requested_model="gpt-4o",
        )

    assert exc_info.value.response.reasoning_content == "partial thought"
    assert exc_info.value.response.content == "partial answer"


async def test_decodes_native_tool_arguments(mocker: MockerFixture) -> None:
    mocker.patch(
        "litellm.stream_chunk_builder",
        return_value=_tool_response("lookup", '{"key":"item"}'),
    )
    events: list[OutputStreamEvent] = []

    async def collect(event: OutputStreamEvent) -> None:
        events.append(event)

    await handle_stream(
        _stream(
            _chunk(tool_name="lookup", tool_arguments='{"key":"'),
            _chunk(tool_arguments='item"}'),
        ),
        content_callback=None,
        output_callback=collect,
        reasoning_callback=None,
        messages=[],
        output=None,
        tool_argument_formats={
            "lookup": StructuredValueFormat.from_generated_schema(
                JsonSchemaDocument.from_mapping(
                    {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                        "additionalProperties": False,
                    }
                )
            )
        },
        requested_model="gpt-4o",
    )

    assert events.count(OutputStringEnd(("tool_calls", 0, "name"), "lookup")) == 1
    assert OutputStringEnd(("tool_calls", 0, "arguments", "key"), "item") in events


async def test_restores_translated_native_tool_arguments(
    mocker: MockerFixture,
) -> None:
    wire_arguments = '{"labels":[{"key":"important","value":2}]}'
    mocker.patch(
        "litellm.stream_chunk_builder",
        return_value=_tool_response("categorize", wire_arguments),
    )
    value_format = StructuredValueFormat.from_generated_schema(
        JsonSchemaDocument.from_mapping(
            {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    }
                },
                "required": ["labels"],
                "additionalProperties": False,
            }
        )
    )
    events: list[OutputStreamEvent] = []

    async def collect(event: OutputStreamEvent) -> None:
        events.append(event)

    response = await handle_stream(
        _stream(
            _chunk(
                tool_name="categorize",
                tool_arguments='{"labels":[{"key":"important",',
            ),
            _chunk(tool_arguments='"value":2}]}'),
        ),
        content_callback=None,
        output_callback=collect,
        reasoning_callback=None,
        messages=[],
        output=None,
        tool_argument_formats={"categorize": value_format},
        requested_model="gpt-4o",
    )

    assert OutputStringEnd(("tool_calls", 0, "name"), "categorize") in events
    assert (
        OutputScalar(("tool_calls", 0, "arguments", "labels", "important"), 2) in events
    )
    assert response.tool_calls[0].arguments.data == {"labels": {"important": 2}}
