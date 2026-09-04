import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Never, Self
from unittest.mock import AsyncMock

import pytest
from litellm import (
    ChatCompletionMessageToolCall,
    Choices,
    Message as LiteLLMMessage,
    ModelResponse,
    Usage,
)
from litellm.exceptions import (
    AuthenticationError,
    InternalServerError,
    RateLimitError,
    Timeout,
)
from litellm.types.utils import (  # pyright: ignore[reportMissingTypeStubs]
    ChatCompletionCustomToolCallPayload,
    ChatCompletionMessageCustomToolCall,
)
from pytest_mock import MockerFixture
from sefia.llm import LLMResponse, Message
from sefia.llm.step_decision import DecisionSpec
from sefia.pydantic import PydanticModelBackend
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
from sefia_litellm._response import handle_response, handle_stream


@pytest.fixture
def mock_acompletion(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch("litellm.acompletion", new_callable=AsyncMock)


@dataclass
class _CityResult:
    city: str


def _decision_model() -> DecisionSpec:
    return DecisionSpec.for_inference(
        output_type=_CityResult,
        tools=[],
        result_format_factory=PydanticModelBackend(),
    )


class TestLiteLLMClient:
    def test_rejects_removed_structured_output_fallback_option(self) -> None:
        with pytest.raises(TypeError, match="PromptedDecisionTransport"):
            LiteLLMClient(model="legacy-model", native_structured_output=False)

    async def test_complete_skips_structured_output_without_decision_model(
        self, mock_acompletion: AsyncMock
    ) -> None:
        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(role="assistant", content="Hi"),
                )
            ]
        )

        await LiteLLMClient(model="gpt-4o").complete(
            [Message(role="user", content="Hello")]
        )

        call_args = mock_acompletion.call_args.kwargs
        assert "response_format" not in call_args
        assert call_args["messages"] == [{"role": "user", "content": "Hello"}]

    async def test_complete_sends_correct_request_to_litellm(
        self, mock_acompletion: AsyncMock
    ):
        client = LiteLLMClient(model="gpt-4o", temperature=0.5)
        messages = [Message(role="user", content="Hello")]
        tools: list[dict[str, Any]] = [
            {"type": "function", "function": {"name": "get_weather"}}
        ]
        decision_model = _decision_model()

        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(role="assistant", content="Hi"),
                )
            ]
        )

        await client.complete(messages, tools=tools, decision_model=decision_model)

        mock_acompletion.assert_called_once()
        call_args = mock_acompletion.call_args[1]
        assert call_args["model"] == "gpt-4o"
        assert call_args["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_args["tools"] == tools
        assert call_args["response_format"]["type"] == "json_schema"
        wire_schema = call_args["response_format"]["json_schema"]["schema"]
        city_schema = wire_schema["properties"]["result"]["properties"]["city"]
        assert city_schema["type"] == "string"
        assert call_args["temperature"] == 0.5

    async def test_complete_does_not_accept_fenced_structured_output(
        self, mock_acompletion: AsyncMock
    ) -> None:
        client = LiteLLMClient(model="gpt-4o")
        model = _decision_model()
        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(
                        role="assistant",
                        content=(
                            '```json\n{"decision":"result",'
                            '"result":{"city":"Tokyo"}}\n```'
                        ),
                    ),
                )
            ]
        )

        response = await client.complete(
            [Message(role="user", content="Follow the task.")],
            decision_model=model,
        )

        call_args = mock_acompletion.call_args.kwargs
        assert call_args["response_format"]["type"] == "json_schema"
        assert call_args["messages"] == [
            {"role": "user", "content": "Follow the task."}
        ]
        assert response.structured_output is None

    async def test_complete_decodes_native_structured_output(
        self, mock_acompletion: AsyncMock
    ) -> None:
        client = LiteLLMClient(model="gpt-4o")
        model = _decision_model()
        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(
                        role="assistant",
                        content=('{"decision":"result","result":{"city":"Tokyo"}}'),
                    ),
                )
            ]
        )

        response = await client.complete(
            [Message(role="user", content="Follow the task.")],
            decision_model=model,
        )

        assert response.structured_output is not None
        assert response.structured_output.data == {
            "decision": "result",
            "result": {"city": "Tokyo"},
        }

    async def test_complete_parses_litellm_response_and_calculates_cost(
        self, mock_acompletion: AsyncMock, mocker: MockerFixture
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
                                function={
                                    "name": "get_weather",
                                    "arguments": '{"city": "Tokyo"}',
                                },
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

    def test_handle_response_rejects_custom_tool_calls(self) -> None:
        response = ModelResponse(
            model="gpt-4o",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=LiteLLMMessage(
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageCustomToolCall(
                                id="call_custom",
                                type="custom",
                                custom=ChatCompletionCustomToolCallPayload(
                                    name="shell", input="pwd"
                                ),
                            )
                        ],
                    ),
                )
            ],
        )

        with pytest.raises(RuntimeError, match="unsupported custom tool call"):
            handle_response(response, requested_model="gpt-4o", output=None)

    async def test_complete_captures_reasoning_content(
        self, mock_acompletion: AsyncMock
    ):
        message = LiteLLMMessage(role="assistant", content='{"decision":...}')
        message.reasoning_content = "The user wants the weather."
        mock_acompletion.return_value = ModelResponse(
            model="gpt-4o",
            choices=[Choices(finish_reason="stop", index=0, message=message)],
        )
        client = LiteLLMClient(model="gpt-4o")

        response = await client.complete([])

        assert response.reasoning_content == "The user wants the weather."

    async def test_cost_is_none_if_calculation_fails(
        self, mock_acompletion: AsyncMock, mocker: MockerFixture
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

    async def test_raises_error_on_empty_choices(self, mock_acompletion: AsyncMock):
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
        self,
        mock_acompletion: AsyncMock,
        provider_error: Exception,
        expected_error: type[Exception],
    ) -> None:
        mock_acompletion.side_effect = provider_error
        client = LiteLLMClient(model="gpt-4o")

        with pytest.raises(expected_error):
            await client.complete([])

    async def test_unmapped_provider_error_propagates_unchanged(
        self, mock_acompletion: AsyncMock
    ):
        mock_acompletion.side_effect = AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4o"
        )
        client = LiteLLMClient(model="gpt-4o")

        with pytest.raises(AuthenticationError):
            await client.complete([])

    async def test_complete_suppresses_litellm_logging_by_default(
        self, mock_acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
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
        self, mock_acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ):
        import litellm

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

        assert litellm.suppress_debug_info is False
        assert logging.getLogger("LiteLLM").level == logging.NOTSET

    def test_env_suppress_logs_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_explicit_suppress_logs_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEFIA_LITELLM_SUPPRESS_LOGS", "false")
        assert LiteLLMClient(model="gpt-4o", suppress_logs=True)._suppress_logs is True
        assert LiteLLMClient(model="gpt-4o")._suppress_logs is False

    async def test_complete_uses_streaming_when_callback_is_provided(
        self, mock_acompletion: AsyncMock, mocker: MockerFixture
    ):
        class FakeStream:
            def __aiter__(self) -> Self:
                return self

            async def __anext__(self) -> Never:
                raise StopAsyncIteration

        stream = FakeStream()
        client = LiteLLMClient(model="gpt-4o")
        stream_response = LLMResponse(content="streamed")
        stream_handler = mocker.patch(
            "sefia_litellm._client.handle_stream",
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
            output=None,
            requested_model="gpt-4o",
        )

    async def test_complete_streams_when_only_reasoning_callback_is_provided(
        self, mock_acompletion: AsyncMock, mocker: MockerFixture
    ):
        class FakeStream:
            def __aiter__(self) -> Self:
                return self

            async def __anext__(self) -> Never:
                raise StopAsyncIteration

        stream = FakeStream()
        client = LiteLLMClient(model="gpt-4o")
        stream_response = LLMResponse(content="streamed")
        stream_handler = mocker.patch(
            "sefia_litellm._client.handle_stream",
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
            output=None,
            requested_model="gpt-4o",
        )

    async def test_handle_stream_routes_reasoning_and_content_separately(
        self, mocker: MockerFixture
    ) -> None:
        def chunk(
            *, content: str | None = None, reasoning: str | None = None
        ) -> SimpleNamespace:
            delta = SimpleNamespace(content=content, reasoning_content=reasoning)
            return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        async def fake_stream() -> AsyncIterator[SimpleNamespace]:
            yield chunk(reasoning="Let me ")
            yield chunk(reasoning="think.")
            yield chunk(content='{"decision"')
            yield chunk(content=":...}")

        built = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(
                        role="assistant", content='{"decision":...}'
                    ),
                )
            ]
        )
        mocker.patch(
            "litellm.stream_chunk_builder",
            return_value=built,
        )

        content_tokens: list[str] = []
        reasoning_tokens: list[str] = []

        async def on_content(token: str) -> None:
            content_tokens.append(token)

        async def on_reasoning(token: str) -> None:
            reasoning_tokens.append(token)

        response = await handle_stream(
            fake_stream(),
            content_callback=on_content,
            output_callback=None,
            reasoning_callback=on_reasoning,
            messages=[],
            output=None,
            requested_model="gpt-4o",
        )

        assert reasoning_tokens == ["Let me ", "think."]
        assert content_tokens == ['{"decision"', ":...}"]
        assert response.reasoning_content == "Let me think."
