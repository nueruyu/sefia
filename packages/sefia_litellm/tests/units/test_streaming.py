from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from litellm import (
    ModelResponse,
)
from pytest_mock import MockerFixture
from sefia.llm import LLMCompletion
from sefia.llm.exceptions import LLMCompletionDecodingError
from sefia.llm.streaming import (
    StringEnd as OutputStringEnd,
)
from sefia_litellm._schema import StructuredDecisionFormat
from sefia_litellm._schema._json_wire_format import JsonWireFormat
from sefia_litellm._streaming import (
    consume_completion_stream,
)

_ResponseFactory = Callable[..., ModelResponse]


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


async def test_routes_reasoning_and_content_separately(
    mocker: MockerFixture,
    make_litellm_response: _ResponseFactory,
) -> None:
    mocker.patch(
        "litellm.stream_chunk_builder",
        return_value=make_litellm_response(content="done"),
    )
    content_tokens: list[str] = []
    reasoning_tokens: list[str] = []

    async def on_content(token: str) -> None:
        content_tokens.append(token)

    async def on_reasoning(token: str) -> None:
        reasoning_tokens.append(token)

    response = await consume_completion_stream(
        _stream(
            _chunk(reasoning="Let me "),
            _chunk(reasoning="think."),
            _chunk(content="answer"),
        ),
        content_callback=on_content,
        output_callback=None,
        reasoning_callback=on_reasoning,
        messages=[],
        decision_format=None,
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
        LLMCompletionDecodingError, match="could not reconstruct"
    ) as exc_info:
        await consume_completion_stream(
            _stream(
                _chunk(reasoning="partial thought"),
                _chunk(content="partial answer"),
            ),
            content_callback=None,
            output_callback=None,
            reasoning_callback=None,
            messages=[],
            decision_format=None,
            requested_model="gpt-4o",
        )

    assert exc_info.value.completion.reasoning_content == "partial thought"
    assert exc_info.value.completion.content == "partial answer"


async def test_stream_dispatches_decoder_events_and_forwards_formats(
    mocker: MockerFixture, make_litellm_response: _ResponseFactory
) -> None:
    native = mocker.patch(
        "sefia_litellm._streaming.NativeToolCallStreamDecoder"
    ).return_value
    parser = mocker.patch(
        "sefia_litellm._streaming.JsonOutputStreamDecoder"
    ).return_value
    event = OutputStringEnd(("payload", "result"), "done")
    logical = OutputStringEnd(("result",), "done")
    final_event = OutputStringEnd(("tool_calls", 0, "name"), "lookup")
    parser.feed.return_value = [event]
    native.feed.return_value = []
    native.finish.return_value = [final_event]
    decision_format = Mock(spec=StructuredDecisionFormat)
    decision_format.decode_stream_event.return_value = logical
    tool_formats: dict[str, JsonWireFormat] = {"lookup": Mock(spec=JsonWireFormat)}
    callback = AsyncMock()
    wire_response = make_litellm_response(content="done")
    builder = mocker.patch("litellm.stream_chunk_builder", return_value=wire_response)
    completion = LLMCompletion(content="done")
    decode = mocker.patch(
        "sefia_litellm._streaming.decode_completion", return_value=completion
    )
    chunks = [_chunk(content="fragment", tool_name="lookup", tool_arguments="{}")]
    messages = [{"role": "user", "content": "prompt"}]

    result = await consume_completion_stream(
        _stream(*chunks),
        content_callback=None,
        output_callback=callback,
        reasoning_callback=None,
        messages=messages,
        decision_format=decision_format,
        tool_data_formats=tool_formats,
        requested_model="gpt-4o",
    )

    assert result is completion
    parser.feed.assert_called_once_with("fragment")
    decision_format.decode_stream_event.assert_called_once_with(event)
    native.feed.assert_called_once_with(chunks[0].choices[0].delta.tool_calls)
    native.finish.assert_called_once_with()
    assert [c.args[0] for c in callback.await_args_list] == [logical, final_event]
    builder.assert_called_once_with(chunks=chunks, messages=messages)
    decode.assert_called_once_with(
        wire_response,
        requested_model="gpt-4o",
        decision_format=decision_format,
        tool_data_formats=tool_formats,
    )
