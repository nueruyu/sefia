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

from sefia.llm.litellm import LiteLLMClient
from sefia.llm.messages import Message


@pytest.fixture
def mock_acompletion(mocker):
    return mocker.patch("sefia.llm.litellm.acompletion", new_callable=AsyncMock)


class TestLiteLLMClient:
    async def test_complete_sends_correct_request_to_litellm(self, mock_acompletion):
        # Arrange
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

        # Act
        await client.complete(messages, tools=tools, output_schema=output_schema)

        # Assert
        mock_acompletion.assert_called_once()
        call_args = mock_acompletion.call_args[1]
        assert call_args["model"] == "gpt-4o"
        assert call_args["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_args["tools"] == tools
        assert call_args["temperature"] == 0.5
        # LiteLLM does not have a direct output_schema param; it's usually
        # handled via response_format or other model-specific kwargs.
        # This test confirms other parameters are passed correctly.

    async def test_complete_parses_litellm_response_correctly(self, mock_acompletion):
        # Arrange
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

        # Act
        response = await client.complete([])

        # Assert
        assert response.model == "gpt-4o"
        assert response.content is None
        assert response.stop_reason == "tool_calls"
        assert response.usage is not None
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.usage["total_tokens"] == 30
        assert len(response.tool_calls) == 1
        tool_call = response.tool_calls[0]
        assert tool_call.id == "call_abc"
        assert tool_call.function["name"] == "get_weather"
        assert tool_call.function["arguments"] == '{"city": "Tokyo"}'

    async def test_raises_error_on_empty_choices(self, mock_acompletion):
        # Arrange
        mock_acompletion.return_value = ModelResponse(choices=[])
        client = LiteLLMClient(model="gpt-4o")

        # Act & Assert
        with pytest.raises(RuntimeError, match="LLM returned empty choices"):
            await client.complete([])
