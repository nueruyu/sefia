import logging
from unittest.mock import AsyncMock

import pytest
from litellm import ModelResponse
from litellm.exceptions import (
    AuthenticationError,
    InternalServerError,
    RateLimitError,
    Timeout,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Usage,
)
from litellm.types.utils import (
    Message as LiteLLMMessage,
)
from pytest_mock import MockerFixture
from sefia.llm import LLMResponse, Message
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


@pytest.fixture
def mock_acompletion(mocker: MockerFixture):
    return mocker.patch("litellm.acompletion", new_callable=AsyncMock)


class TestLiteLLMClient:
    async def test_complete_sends_correct_request_to_litellm(self, mock_acompletion):
        client = LiteLLMClient(model="gpt-4o", temperature=0.5)
        messages = [Message(role="user", content="Hello")]
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        output_schema = {"type": "object", "properties": {"city": {"type": "string"}}}

        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(role="assistant", content="Hi"),
                )
            ]
        )

        await client.complete(messages, tools=tools, output_schema=output_schema)

        mock_acompletion.assert_called_once()
        call_args = mock_acompletion.call_args[1]
        assert call_args["model"] == "gpt-4o"
        assert call_args["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_args["tools"] == tools
        assert call_args["response_format"]["type"] == "json_schema"
        assert call_args["response_format"]["json_schema"]["schema"] == output_schema
        assert call_args["temperature"] == 0.5

    async def test_complete_parses_litellm_response_and_calculates_cost(
        self, mock_acompletion, mocker: MockerFixture
    ):
        mocker.patch("litellm.cost_per_token", return_value=(0.001, 0.002))
        mock_response = ModelResponse(
            id="chatcmpl-123",
            model="gpt-4o",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=LiteLLMMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call_abc",
                                function=Function(
                                    name="get_weather", arguments='{"city": "Tokyo"}'
                                ),
                                type="function",
                            )
                        ],
                    ),
                )
            ],
        )
        mock_acompletion.return_value = mock_response
        client = LiteLLMClient(model="gpt-4o")

        response = await client.complete([])

        assert response.model == "gpt-4o"
        assert response.cost == 0.003
        assert response.content is None
        assert response.stop_reason == "tool_calls"
        assert response.usage is not None
        assert response.usage["prompt_tokens"] == 10
        assert len(response.tool_calls) == 1

    async def test_cost_is_none_if_calculation_fails(
        self, mock_acompletion, mocker: MockerFixture
    ):
        mocker.patch("litellm.cost_per_token", side_effect=Exception("API error"))
        mock_response = ModelResponse(
            model="gpt-4o",
            usage=Usage(prompt_tokens=10, completion_tokens=20),
            choices=[Choices(index=0, message=LiteLLMMessage(role="assistant"))],
        )
        mock_acompletion.return_value = mock_response
        client = LiteLLMClient(model="gpt-4o")

        response = await client.complete([])

        assert response.cost is None

    async def test_raises_error_on_empty_choices(self, mock_acompletion):
        mock_acompletion.return_value = ModelResponse(choices=[])
        client = LiteLLMClient(model="gpt-4o")

        with pytest.raises(RuntimeError, match="LLM returned empty choices"):
            await client.complete([])

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
                InternalServerError(
                    message="boom", llm_provider="openai", model="gpt-4o"
                ),
                InferenceTemporarilyUnavailableError,
            ),
        ],
    )
    async def test_provider_errors_map_to_inference_errors(
        self, mock_acompletion, provider_error, expected_error
    ):
        # The adapter translates LiteLLM's transient errors into sefia's abstract
        # errors so callers never have to know about LiteLLM's types.
        mock_acompletion.side_effect = provider_error
        client = LiteLLMClient(model="gpt-4o")

        with pytest.raises(expected_error):
            await client.complete([])

    async def test_unmapped_provider_error_propagates_unchanged(self, mock_acompletion):
        # A deterministic error (bad credentials) is not mapped; it surfaces as
        # itself so it is engraved as a genuine failure.
        mock_acompletion.side_effect = AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4o"
        )
        client = LiteLLMClient(model="gpt-4o")

        with pytest.raises(AuthenticationError):
            await client.complete([])

    async def test_complete_suppresses_litellm_logging_by_default(
        self, mock_acompletion, monkeypatch
    ):
        import litellm

        monkeypatch.setattr(litellm, "suppress_debug_info", False, raising=False)
        logging.getLogger("LiteLLM").setLevel(logging.NOTSET)
        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(role="assistant", content="Hi"),
                )
            ]
        )
        client = LiteLLMClient(model="gpt-4o")

        await client.complete([])

        assert litellm.suppress_debug_info is True
        assert logging.getLogger("LiteLLM").level == _SILENCE_LEVEL

    async def test_complete_restores_litellm_logging_when_disabled(
        self, mock_acompletion, monkeypatch
    ):
        import litellm

        # Simulate an earlier client having suppressed logging globally.
        monkeypatch.setattr(litellm, "suppress_debug_info", True, raising=False)
        logging.getLogger("LiteLLM").setLevel(_SILENCE_LEVEL)
        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(role="assistant", content="Hi"),
                )
            ]
        )
        client = LiteLLMClient(model="gpt-4o", suppress_logs=False)

        await client.complete([])

        # suppress_logs=False wins over the earlier suppression and restores defaults.
        assert litellm.suppress_debug_info is False
        assert logging.getLogger("LiteLLM").level == logging.NOTSET

    def test_env_suppress_logs_default(self, monkeypatch):
        monkeypatch.delenv("SEFIA_LITELLM_SUPPRESS_LOGS", raising=False)
        assert _env_suppress_logs_default() is True

        monkeypatch.setenv("SEFIA_LITELLM_SUPPRESS_LOGS", "false")
        assert _env_suppress_logs_default() is False

        monkeypatch.setenv("SEFIA_LITELLM_SUPPRESS_LOGS", "1")
        assert _env_suppress_logs_default() is True

    def test_apply_litellm_log_level(self):
        lg = logging.getLogger("LiteLLM")
        _apply_litellm_log_level(True)
        assert lg.level == _SILENCE_LEVEL
        _apply_litellm_log_level(False)
        assert lg.level == logging.NOTSET

    def test_explicit_suppress_logs_overrides_env(self, monkeypatch):
        monkeypatch.setenv("SEFIA_LITELLM_SUPPRESS_LOGS", "false")
        assert LiteLLMClient(model="gpt-4o", suppress_logs=True)._suppress_logs is True
        assert LiteLLMClient(model="gpt-4o")._suppress_logs is False

    async def test_complete_uses_streaming_when_callback_is_provided(
        self, mock_acompletion, mocker
    ):
        class FakeStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        stream = FakeStream()
        client = LiteLLMClient(model="gpt-4o")
        stream_response = LLMResponse(content="streamed")
        handle_stream = mocker.patch.object(
            client,
            "_handle_stream",
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
        handle_stream.assert_awaited_once_with(
            stream,
            callback,
            [{"role": "user", "content": "Hello"}],
        )
