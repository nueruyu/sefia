from unittest.mock import AsyncMock

import pytest
from litellm import ModelResponse
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
from sefia.llm.messages import LLMResponse, Message
from sefia_litellm.client import LiteLLMClient


@pytest.fixture
def mock_acompletion(mocker: MockerFixture):
    return mocker.patch("sefia_litellm.client.acompletion", new_callable=AsyncMock)


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
        mocker.patch("sefia_litellm.client.cost_per_token", return_value=(0.001, 0.002))
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
        mocker.patch(
            "sefia_litellm.client.cost_per_token", side_effect=Exception("API error")
        )
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
