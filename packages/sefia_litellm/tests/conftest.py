"""Shared LiteLLM test fixtures."""

from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest
from litellm import (
    ChatCompletionMessageToolCall,
    Choices,
    Message as LiteLLMMessage,
    ModelResponse,
)
from pytest_mock import MockerFixture


@pytest.fixture
def mock_acompletion(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch("litellm.acompletion", new_callable=AsyncMock)


@pytest.fixture
def make_litellm_response() -> Callable[..., ModelResponse]:
    def factory(
        *,
        content: str | None = None,
        finish_reason: str = "stop",
        tool_calls: list[ChatCompletionMessageToolCall] | None = None,
        model: str | None = None,
    ) -> ModelResponse:
        choices = [
            Choices(
                finish_reason=finish_reason,
                index=0,
                message=LiteLLMMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                ),
            )
        ]
        if model is None:
            return ModelResponse(choices=choices)
        return ModelResponse(model=model, choices=choices)

    return factory
