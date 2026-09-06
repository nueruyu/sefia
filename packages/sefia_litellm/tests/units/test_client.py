import logging
from collections.abc import Callable
from typing import Never, Self
from unittest.mock import AsyncMock

import pytest
from litellm import ModelResponse
from litellm.exceptions import (
    AuthenticationError,
    InternalServerError,
    RateLimitError,
    Timeout,
)
from pytest_mock import MockerFixture
from sefia.llm import (
    LLMCompletion,
    Message,
)
from sefia.llm.exceptions import LLMCompletionDecodingError
from sefia_litellm._client import (
    _SILENCE_LEVEL,
    LiteLLMClient,
    _apply_litellm_log_level,
    _env_suppress_logs_default,
)
from sefia_litellm.exceptions import (
    InferenceRateLimitError,
    InferenceTemporarilyUnavailableError,
    InferenceTimeoutError,
)

_ResponseFactory = Callable[..., ModelResponse]


class _EmptyStream:
    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Never:
        raise StopAsyncIteration


def test_rejects_removed_structured_output_fallback_option() -> None:
    with pytest.raises(TypeError, match="PromptedDecisionTransport"):
        LiteLLMClient(model="legacy-model", native_structured_output=False)


@pytest.mark.parametrize(
    ("provider_error", "expected_error"),
    [
        (
            RateLimitError(
                message="rate limited", llm_provider="openai", model="gpt-4o"
            ),
            InferenceRateLimitError,
        ),
        (
            Timeout(message="timed out", model="gpt-4o", llm_provider="openai"),
            InferenceTimeoutError,
        ),
        (
            InternalServerError(message="boom", llm_provider="openai", model="gpt-4o"),
            InferenceTemporarilyUnavailableError,
        ),
    ],
)
async def test_provider_errors_map_to_inference_errors(
    mock_acompletion: AsyncMock,
    provider_error: Exception,
    expected_error: type[Exception],
) -> None:
    mock_acompletion.side_effect = provider_error
    client = LiteLLMClient(model="gpt-4o")

    with pytest.raises(expected_error):
        await client.complete([])


async def test_unmapped_provider_error_propagates_unchanged(
    mock_acompletion: AsyncMock,
):
    mock_acompletion.side_effect = AuthenticationError(
        message="bad key", llm_provider="openai", model="gpt-4o"
    )
    client = LiteLLMClient(model="gpt-4o")

    with pytest.raises(AuthenticationError):
        await client.complete([])


async def test_complete_suppresses_litellm_logging_by_default(
    mock_acompletion: AsyncMock,
    make_litellm_response: _ResponseFactory,
    monkeypatch: pytest.MonkeyPatch,
):
    import litellm

    monkeypatch.setattr(litellm, "suppress_debug_info", False, raising=False)
    logging.getLogger("LiteLLM").setLevel(logging.NOTSET)
    mock_acompletion.return_value = make_litellm_response(content="Hi")
    client = LiteLLMClient(model="gpt-4o")

    await client.complete([])

    assert litellm.suppress_debug_info is True
    assert logging.getLogger("LiteLLM").level == _SILENCE_LEVEL


async def test_complete_restores_litellm_logging_when_disabled(
    mock_acompletion: AsyncMock,
    make_litellm_response: _ResponseFactory,
    monkeypatch: pytest.MonkeyPatch,
):
    import litellm

    monkeypatch.setattr(litellm, "suppress_debug_info", True, raising=False)
    logging.getLogger("LiteLLM").setLevel(_SILENCE_LEVEL)
    mock_acompletion.return_value = make_litellm_response(content="Hi")
    client = LiteLLMClient(model="gpt-4o", suppress_logs=False)

    await client.complete([])

    assert litellm.suppress_debug_info is False
    assert logging.getLogger("LiteLLM").level == logging.NOTSET


async def test_explicit_suppression_overrides_disabled_environment(
    mock_acompletion: AsyncMock,
    make_litellm_response: _ResponseFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    monkeypatch.setenv("SEFIA_LITELLM_SUPPRESS_LOGS", "false")
    monkeypatch.setattr(litellm, "suppress_debug_info", False, raising=False)
    logging.getLogger("LiteLLM").setLevel(logging.NOTSET)
    mock_acompletion.return_value = make_litellm_response(content="Hi")

    await LiteLLMClient(model="gpt-4o", suppress_logs=True).complete([])

    assert litellm.suppress_debug_info is True
    assert logging.getLogger("LiteLLM").level == _SILENCE_LEVEL


def test_env_suppress_logs_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEFIA_LITELLM_SUPPRESS_LOGS", raising=False)
    assert _env_suppress_logs_default() is True

    monkeypatch.setenv("SEFIA_LITELLM_SUPPRESS_LOGS", "false")
    assert _env_suppress_logs_default() is False

    monkeypatch.setenv("SEFIA_LITELLM_SUPPRESS_LOGS", "1")
    assert _env_suppress_logs_default() is True


def test_apply_litellm_log_level():
    lg = logging.getLogger("LiteLLM")
    _apply_litellm_log_level(True)
    assert lg.level == _SILENCE_LEVEL
    _apply_litellm_log_level(False)
    assert lg.level == logging.NOTSET


async def test_complete_uses_streaming_when_callback_is_provided(
    mock_acompletion: AsyncMock, mocker: MockerFixture
):
    stream = _EmptyStream()
    client = LiteLLMClient(model="gpt-4o")
    stream_response = LLMCompletion(content="streamed")
    stream_handler = mocker.patch(
        "sefia_litellm._client.consume_completion_stream",
        new_callable=AsyncMock,
        return_value=stream_response,
    )
    callback = AsyncMock()
    messages = [Message(role="user", content="Hello")]

    mock_acompletion.return_value = stream

    response = await client.complete(messages, stream_callback=callback)

    assert response == stream_response
    call_args = mock_acompletion.call_args[1]
    assert call_args["stream"] is True
    stream_handler.assert_awaited_once_with(
        stream,
        content_callback=callback,
        output_callback=None,
        reasoning_callback=None,
        messages=[{"role": "user", "content": "Hello"}],
        decision_format=None,
        tool_data_formats={},
        requested_model="gpt-4o",
    )


async def test_complete_streams_when_only_reasoning_callback_is_provided(
    mock_acompletion: AsyncMock, mocker: MockerFixture
):
    stream = _EmptyStream()
    client = LiteLLMClient(model="gpt-4o")
    stream_response = LLMCompletion(content="streamed")
    stream_handler = mocker.patch(
        "sefia_litellm._client.consume_completion_stream",
        new_callable=AsyncMock,
        return_value=stream_response,
    )
    reasoning_callback = AsyncMock()
    messages = [Message(role="user", content="Hello")]
    mock_acompletion.return_value = stream

    await client.complete(messages, reasoning_callback=reasoning_callback)

    assert mock_acompletion.call_args[1]["stream"] is True
    stream_handler.assert_awaited_once_with(
        stream,
        content_callback=None,
        output_callback=None,
        reasoning_callback=reasoning_callback,
        messages=[{"role": "user", "content": "Hello"}],
        decision_format=None,
        tool_data_formats={},
        requested_model="gpt-4o",
    )


async def test_complete_streams_when_only_output_callback_is_provided(
    mock_acompletion: AsyncMock, mocker: MockerFixture
) -> None:
    stream = _EmptyStream()
    client = LiteLLMClient(model="gpt-4o")
    stream_response = LLMCompletion(content="streamed")
    stream_handler = mocker.patch(
        "sefia_litellm._client.consume_completion_stream",
        new_callable=AsyncMock,
        return_value=stream_response,
    )
    output_callback = AsyncMock()
    messages = [Message(role="user", content="Hello")]
    mock_acompletion.return_value = stream

    await client.complete(messages, output_callback=output_callback)

    assert mock_acompletion.call_args[1]["stream"] is True
    stream_handler.assert_awaited_once_with(
        stream,
        content_callback=None,
        output_callback=output_callback,
        reasoning_callback=None,
        messages=[{"role": "user", "content": "Hello"}],
        decision_format=None,
        tool_data_formats={},
        requested_model="gpt-4o",
    )


async def test_complete_rejects_non_litellm_completion(
    mock_acompletion: AsyncMock,
) -> None:
    mock_acompletion.return_value = object()

    with pytest.raises(
        LLMCompletionDecodingError, match="unsupported completion response"
    ) as exc_info:
        await LiteLLMClient(model="gpt-4o").complete(
            [Message(role="user", content="Hello")]
        )

    assert exc_info.value.completion.model == "gpt-4o"
