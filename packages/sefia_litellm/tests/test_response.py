import pytest
from litellm import (
    ChatCompletionMessageToolCall,
    Choices,
    Message as LiteLLMMessage,
    ModelResponse,
    Usage,
)
from litellm.types.utils import (  # pyright: ignore[reportMissingTypeStubs]
    ChatCompletionCustomToolCallPayload,
    ChatCompletionMessageCustomToolCall,
)
from pytest_mock import MockerFixture
from sefia.llm import LLMOutput, ToolCall
from sefia.llm.exceptions import LLMResponseDecodingError
from sefia_litellm._response import handle_response


def test_converts_response_and_calculates_cost(mocker: MockerFixture) -> None:
    mocker.patch("litellm.cost_per_token", return_value=(0.001, 0.002))
    response = ModelResponse(
        model="gpt-4o",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=LiteLLMMessage(
                    role="assistant",
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

    result = handle_response(response, requested_model="gpt-4o", output=None)

    assert result.model == "gpt-4o"
    assert result.cost == 0.003
    assert result.content is None
    assert result.stop_reason == "tool_calls"
    assert result.usage is not None
    assert result.usage["prompt_tokens"] == 10
    assert result.tool_calls == [
        ToolCall(
            id="call_abc",
            name="get_weather",
            arguments=LLMOutput.from_json({"city": "Tokyo"}),
        )
    ]


def test_rejects_custom_tool_calls_with_partial_response() -> None:
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

    with pytest.raises(
        LLMResponseDecodingError,
        match="unsupported custom tool call",
    ) as exc_info:
        handle_response(response, requested_model="gpt-4o", output=None)

    assert exc_info.value.response.model == "gpt-4o"


def test_rejects_malformed_tool_arguments() -> None:
    response = ModelResponse(
        model="gpt-4o",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=LiteLLMMessage(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call-1",
                            function={"name": "lookup", "arguments": "not json"},
                            type="function",
                        )
                    ],
                ),
            )
        ],
    )

    with pytest.raises(LLMResponseDecodingError):
        handle_response(response, requested_model="gpt-4o", output=None)


def test_preserves_reasoning_content() -> None:
    message = LiteLLMMessage(role="assistant", content="done")
    message.reasoning_content = "The user wants the weather."
    response = ModelResponse(
        model="gpt-4o",
        choices=[Choices(finish_reason="stop", index=0, message=message)],
    )

    result = handle_response(response, requested_model="gpt-4o", output=None)

    assert result.reasoning_content == "The user wants the weather."


def test_cost_is_none_if_calculation_fails(mocker: MockerFixture) -> None:
    mocker.patch("litellm.cost_per_token", side_effect=Exception("API error"))
    response = ModelResponse(
        model="gpt-4o",
        usage=Usage(prompt_tokens=10, completion_tokens=20),
        choices=[Choices(index=0, message=LiteLLMMessage(role="assistant"))],
    )

    result = handle_response(response, requested_model="gpt-4o", output=None)

    assert result.cost is None


def test_empty_choices_are_a_response_decoding_error() -> None:
    response = ModelResponse(model="provider-model", choices=[])

    with pytest.raises(
        LLMResponseDecodingError, match="LLM returned empty choices"
    ) as exc_info:
        handle_response(response, requested_model="requested-model", output=None)

    assert exc_info.value.response.model == "provider-model"
