import pytest
from glyff.exceptions import YieldException
from sefia import InferenceContext
from sefia.exceptions import InferenceTimeoutError, SefiaError
from sefios.middleware import (
    MaxRetriesExceededError,
    MaxStepsExceededError,
    Retrier,
)


def _ctx() -> InferenceContext:
    return InferenceContext(func_name="f", args=(), kwargs={})


class TestRetrier:
    def test_rejects_negative_max_retries(self):
        with pytest.raises(ValueError):
            Retrier(max_retries=-1)

    async def test_returns_result_on_success(self):
        middleware = Retrier(max_retries=3)

        async def nxt():
            return "ok"

        assert await middleware.wrap(_ctx(), nxt) == "ok"

    async def test_retries_until_success_within_limits(self):
        middleware = Retrier(max_retries=3)
        calls = 0

        async def flaky():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ValueError("boom")
            return "ok"

        assert await middleware.wrap(_ctx(), flaky) == "ok"
        assert calls == 3

    async def test_raises_error_when_retries_exceeded(self):
        middleware = Retrier(max_retries=2)
        calls = 0

        async def failing():
            nonlocal calls
            calls += 1
            raise ValueError("boom")

        with pytest.raises(MaxRetriesExceededError):
            await middleware.wrap(_ctx(), failing)
        assert calls == 3

    async def test_preserves_original_error_as_cause(self):
        middleware = Retrier(max_retries=0)
        original = ValueError("boom")

        async def failing():
            raise original

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            await middleware.wrap(_ctx(), failing)
        assert exc_info.value.__cause__ is original

    async def test_does_not_retry_terminal_exceptions(self):
        middleware = Retrier(max_retries=3)

        async def hit_limit():
            raise MaxStepsExceededError("stop")

        with pytest.raises(MaxStepsExceededError):
            await middleware.wrap(_ctx(), hit_limit)

    async def test_does_not_retry_sefia_errors(self):
        middleware = Retrier(max_retries=3)
        calls = 0

        async def fail_with_framework_error():
            nonlocal calls
            calls += 1
            raise SefiaError("stop")

        with pytest.raises(SefiaError):
            await middleware.wrap(_ctx(), fail_with_framework_error)
        assert calls == 1

    async def test_retries_inference_errors(self):
        middleware = Retrier(max_retries=1)
        calls = 0

        async def fail_then_succeed():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise InferenceTimeoutError("timeout")
            return "ok"

        assert await middleware.wrap(_ctx(), fail_then_succeed) == "ok"
        assert calls == 2

    async def test_does_not_retry_yield_exception(self):
        middleware = Retrier(max_retries=3)

        async def interrupt():
            raise YieldException("resume later")

        with pytest.raises(YieldException):
            await middleware.wrap(_ctx(), interrupt)
