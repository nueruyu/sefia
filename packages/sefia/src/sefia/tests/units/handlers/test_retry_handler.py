import pytest

from sefia.events import AttemptStart, ToolExecutionFailed
from sefia.handlers.retry import (
    MaxRetriesExceededError,
    RequestInferenceRetry,
    RetryHandler,
)
from sefia.models import ToolCallRequest


class TestRetryHandler:
    async def test_requests_retry_within_limits(self):
        handler = RetryHandler(max_retries=3)
        tool_error = ToolExecutionFailed(
            tool_call=ToolCallRequest(id="1", name="t", arguments={}),
            error=ValueError(),
        )

        await handler.handle(AttemptStart())  # Attempt 1
        with pytest.raises(RequestInferenceRetry):
            await handler.handle(tool_error)

        await handler.handle(AttemptStart())  # Attempt 2
        with pytest.raises(RequestInferenceRetry):
            await handler.handle(tool_error)

        await handler.handle(AttemptStart())  # Attempt 3
        with pytest.raises(RequestInferenceRetry):
            await handler.handle(tool_error)

    async def test_raises_error_when_retries_exceeded(self):
        handler = RetryHandler(max_retries=2)
        tool_error = ToolExecutionFailed(
            tool_call=ToolCallRequest(id="1", name="t", arguments={}),
            error=ValueError(),
        )

        await handler.handle(AttemptStart())  # Attempt 1
        with pytest.raises(RequestInferenceRetry):
            await handler.handle(tool_error)  # Retry 1

        await handler.handle(AttemptStart())  # Attempt 2
        with pytest.raises(RequestInferenceRetry):
            await handler.handle(tool_error)  # Retry 2

        await handler.handle(AttemptStart())  # Attempt 3
        with pytest.raises(MaxRetriesExceededError):
            await handler.handle(tool_error)  # Exceeded

    async def test_custom_max_retries_is_respected(self):
        handler = RetryHandler(max_retries=1)
        tool_error = ToolExecutionFailed(
            tool_call=ToolCallRequest(id="1", name="t", arguments={}),
            error=ValueError(),
        )

        await handler.handle(AttemptStart())  # Attempt 1
        with pytest.raises(RequestInferenceRetry):
            await handler.handle(tool_error)

        with pytest.raises(MaxRetriesExceededError):
            await handler.handle(tool_error)
