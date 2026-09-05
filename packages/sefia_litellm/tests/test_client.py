import logging
from dataclasses import dataclass
from typing import Never, Self
from unittest.mock import AsyncMock

import pytest
from litellm import (
    ChatCompletionMessageToolCall,
    Choices,
    Message as LiteLLMMessage,
    ModelResponse,
)
from litellm.exceptions import (
    AuthenticationError,
    InternalServerError,
    RateLimitError,
    Timeout,
)
from pytest_mock import MockerFixture
from sefia._tool_system import ToolRegistry
from sefia.llm import (
    LLMCompletion,
    Message,
    ToolCall,
)
from sefia.llm.exceptions import LLMCompletionDecodingError
from sefia.llm.json_schema import JsonSchemaDocument
from sefia.llm.structured_data import StructuredData
from sefia.llm.step_decision import DecisionSpec, StepTool, ToolSchemaSource
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


@pytest.fixture
def mock_acompletion(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch("litellm.acompletion", new_callable=AsyncMock)


@dataclass
class _CityResult:
    city: str


def _decision_spec() -> DecisionSpec:
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

    async def test_complete_rejects_non_litellm_completion(
        self, mock_acompletion: AsyncMock
    ) -> None:
        mock_acompletion.return_value = object()

        with pytest.raises(
            LLMCompletionDecodingError, match="unsupported completion response"
        ) as exc_info:
            await LiteLLMClient(model="gpt-4o").complete(
                [Message(role="user", content="Hello")]
            )

        assert exc_info.value.completion.model == "gpt-4o"

    async def test_complete_sends_correct_request_to_litellm(
        self, mock_acompletion: AsyncMock
    ):
        client = LiteLLMClient(model="gpt-4o", temperature=0.5)
        messages = [Message(role="user", content="Hello")]
        tools = [
            StepTool(
                name="get_weather",
                description="",
                arguments=JsonSchemaDocument.from_mapping(
                    {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    }
                ),
                schema_source=ToolSchemaSource.GENERATED,
            )
        ]
        decision_spec = _decision_spec()

        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(role="assistant", content="Hi"),
                )
            ]
        )

        await client.complete(messages, tools=tools, decision_spec=decision_spec)

        mock_acompletion.assert_called_once()
        call_args = mock_acompletion.call_args[1]
        assert call_args["model"] == "gpt-4o"
        assert call_args["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_args["tools"][0]["function"] == {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
        assert call_args["response_format"]["type"] == "json_schema"
        wire_schema = call_args["response_format"]["json_schema"]["schema"]
        city_schema = wire_schema["properties"]["payload"]["properties"]["result"][
            "properties"
        ]["city"]
        assert city_schema["type"] == "string"
        assert call_args["temperature"] == 0.5

    async def test_complete_does_not_accept_fenced_structured_output(
        self, mock_acompletion: AsyncMock
    ) -> None:
        client = LiteLLMClient(model="gpt-4o")
        decision_spec = _decision_spec()
        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(
                        role="assistant",
                        content=(
                            '```json\n{"payload":{"decision":"result",'
                            '"result":{"city":"Tokyo"}}}\n```'
                        ),
                    ),
                )
            ]
        )

        response = await client.complete(
            [Message(role="user", content="Follow the task.")],
            decision_spec=decision_spec,
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
        decision_spec = _decision_spec()
        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(
                        role="assistant",
                        content=(
                            '{"payload":{"decision":"result",'
                            '"result":{"city":"Tokyo"}}}'
                        ),
                    ),
                )
            ]
        )

        response = await client.complete(
            [Message(role="user", content="Follow the task.")],
            decision_spec=decision_spec,
        )

        assert response.structured_output is not None
        assert response.structured_output.tree == {
            "decision": "result",
            "result": {"city": "Tokyo"},
        }

    async def test_complete_envelopes_tool_or_result_union(
        self, mock_acompletion: AsyncMock
    ) -> None:
        def lookup(key: str) -> str:
            return key

        registry = ToolRegistry()
        registry.add(lookup, name="lookup")
        decision_spec = DecisionSpec.for_inference(
            output_type=str,
            tools=registry.get_all(),
            result_format_factory=PydanticModelBackend(),
        )
        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(
                        role="assistant",
                        content=(
                            '{"payload":{"decision":"tool_calls","tool_calls":['
                            '{"name":"lookup","arguments":{"key":"item"}}]}}'
                        ),
                    ),
                )
            ]
        )

        response = await LiteLLMClient(model="gpt-4o").complete(
            [Message(role="user", content="Look up the item.")],
            decision_spec=decision_spec,
        )

        schema = mock_acompletion.call_args.kwargs["response_format"]["json_schema"][
            "schema"
        ]
        assert schema["type"] == "object"
        assert "anyOf" not in schema
        assert "anyOf" in schema["properties"]["payload"]
        assert response.structured_output is not None
        assert response.structured_output.tree == {
            "decision": "tool_calls",
            "tool_calls": [{"name": "lookup", "arguments": {"key": "item"}}],
        }

    async def test_complete_translates_native_tool_schema_and_arguments(
        self,
        mock_acompletion: AsyncMock,
    ) -> None:
        tool = StepTool(
            name="categorize",
            description="Categorize labels.",
            arguments=JsonSchemaDocument.from_mapping(
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
            ),
            schema_source=ToolSchemaSource.GENERATED,
        )
        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=LiteLLMMessage(
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id="call-1",
                                function={
                                    "name": "categorize",
                                    "arguments": (
                                        '{"labels":[{"key":"important","value":2}]}'
                                    ),
                                },
                                type="function",
                            )
                        ],
                    ),
                )
            ]
        )

        response = await LiteLLMClient(model="gpt-4o").complete(
            [Message(role="user", content="Categorize.")],
            tools=[tool],
        )

        sent_schema = mock_acompletion.call_args.kwargs["tools"][0]["function"][
            "parameters"
        ]
        assert sent_schema["properties"]["labels"]["type"] == "array"
        assert response.tool_calls[0].arguments.tree == {"labels": {"important": 2}}

    async def test_complete_encodes_native_tool_call_history_for_wire_schema(
        self,
        mock_acompletion: AsyncMock,
    ) -> None:
        tool = StepTool(
            name="categorize",
            description="",
            arguments=JsonSchemaDocument.from_mapping(
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
            ),
            schema_source=ToolSchemaSource.GENERATED,
        )
        mock_acompletion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=LiteLLMMessage(role="assistant", content="done"),
                )
            ]
        )
        messages = [
            Message(role="user", content="Categorize."),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="categorize",
                        arguments=StructuredData.from_json(
                            {"labels": {"important": 2}}
                        ),
                    )
                ],
            ),
            Message(role="tool", content="done", tool_call_id="call-1"),
        ]

        await LiteLLMClient(model="gpt-4o").complete(messages, tools=[tool])

        sent_call = mock_acompletion.call_args.kwargs["messages"][1]["tool_calls"][0]
        assert sent_call == {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "categorize",
                "arguments": ('{"labels":[{"key":"important","value":2}]}'),
            },
        }

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
        self, mock_acompletion: AsyncMock, mocker: MockerFixture
    ):
        class FakeStream:
            def __aiter__(self) -> Self:
                return self

            async def __anext__(self) -> Never:
                raise StopAsyncIteration

        stream = FakeStream()
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
