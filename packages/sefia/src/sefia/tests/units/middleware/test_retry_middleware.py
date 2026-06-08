import pytest
from glyff.exceptions import YieldException

from sefia.interfaces.middleware import RunContext
from sefia.middleware.retry import RetryMiddleware
from sefia.middleware.signals import (
    MaxRetriesExceededError,
    MaxStepsExceededError,
    RequestInferenceRetry,
)


def _ctx() -> RunContext:
    return RunContext(func_name="f", args=(), kwargs={})


class TestRetryMiddleware:
    def test_rejects_negative_max_retries(self):
        with pytest.raises(ValueError):
            RetryMiddleware(max_retries=-1)

    async def test_returns_result_on_success(self):
        middleware = RetryMiddleware(max_retries=3)

        async def nxt():
            return "ok"

        assert await middleware.wrap(_ctx(), nxt) == "ok"

    async def test_requests_retry_within_limits(self):
        middleware = RetryMiddleware(max_retries=3)

        async def failing():
            raise ValueError("boom")

        for _ in range(3):
            with pytest.raises(RequestInferenceRetry):
                await middleware.wrap(_ctx(), failing)

    async def test_raises_error_when_retries_exceeded(self):
        middleware = RetryMiddleware(max_retries=2)

        async def failing():
            raise ValueError("boom")

        with pytest.raises(RequestInferenceRetry):
            await middleware.wrap(_ctx(), failing)
        with pytest.raises(RequestInferenceRetry):
            await middleware.wrap(_ctx(), failing)
        with pytest.raises(MaxRetriesExceededError):
            await middleware.wrap(_ctx(), failing)

    async def test_preserves_original_error_as_cause(self):
        middleware = RetryMiddleware(max_retries=0)
        original = ValueError("boom")

        async def failing():
            raise original

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            await middleware.wrap(_ctx(), failing)
        assert exc_info.value.__cause__ is original

    async def test_does_not_retry_terminal_control_signals(self):
        # Terminal limits (e.g. max steps) must not be retried away.
        middleware = RetryMiddleware(max_retries=3)

        async def hit_limit():
            raise MaxStepsExceededError("stop")

        with pytest.raises(MaxStepsExceededError):
            await middleware.wrap(_ctx(), hit_limit)

    async def test_does_not_retry_yield_exception(self):
        # YieldException is a graceful, resumable interrupt and must propagate.
        middleware = RetryMiddleware(max_retries=3)

        async def interrupt():
            raise YieldException("resume later")

        with pytest.raises(YieldException):
            await middleware.wrap(_ctx(), interrupt)
