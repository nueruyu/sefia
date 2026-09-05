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
from sefia.llm import ToolCall
from sefia.llm.exceptions import LLMCompletionDecodingError
from sefia.llm.structured_data import StructuredData
from sefia_litellm._response import decode_completion


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

    completion = decode_completion(
        response, requested_model="gpt-4o", decision_format=None
    )

    assert completion.model == "gpt-4o"
    assert completion.cost == 0.003
    assert completion.content is None
    assert completion.stop_reason == "tool_calls"
    assert completion.usage is not None
    assert completion.usage["prompt_tokens"] == 10
    assert completion.tool_calls == [
        ToolCall(
            id="call_abc",
            name="get_weather",
            arguments=StructuredData.from_json({"city": "Tokyo"}),
        )
    ]


def test_rejects_custom_tool_calls_with_partial_completion() -> None:
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
        LLMCompletionDecodingError,
        match="unsupported custom tool call",
    ) as exc_info:
        decode_completion(response, requested_model="gpt-4o", decision_format=None)

    assert exc_info.value.completion.model == "gpt-4o"


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

    with pytest.raises(LLMCompletionDecodingError):
        decode_completion(response, requested_model="gpt-4o", decision_format=None)


def test_preserves_reasoning_content() -> None:
    message = LiteLLMMessage(role="assistant", content="done")
    message.reasoning_content = "The user wants the weather."
    response = ModelResponse(
        model="gpt-4o",
        choices=[Choices(finish_reason="stop", index=0, message=message)],
    )

    completion = decode_completion(
        response, requested_model="gpt-4o", decision_format=None
    )

    assert completion.reasoning_content == "The user wants the weather."


def test_cost_is_none_if_calculation_fails(mocker: MockerFixture) -> None:
    mocker.patch("litellm.cost_per_token", side_effect=Exception("API error"))
    response = ModelResponse(
        model="gpt-4o",
        usage=Usage(prompt_tokens=10, completion_tokens=20),
        choices=[Choices(index=0, message=LiteLLMMessage(role="assistant"))],
    )

    completion = decode_completion(
        response, requested_model="gpt-4o", decision_format=None
    )

    assert completion.cost is None


def test_empty_choices_are_a_completion_decoding_error() -> None:
    response = ModelResponse(model="provider-model", choices=[])

    with pytest.raises(
        LLMCompletionDecodingError, match="LLM returned empty choices"
    ) as exc_info:
        decode_completion(
            response, requested_model="requested-model", decision_format=None
        )

    assert exc_info.value.completion.model == "provider-model"
