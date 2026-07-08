import pytest
from sefia.exceptions import PauseException
from sefia import InferenceContext
from sefia.exceptions import InvalidInferenceResponseError, SefiaError
from sefios.middleware import (
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

    async def test_does_not_retry_genuine_exceptions(self):
        # A non-recoverable error (anything that is not an InferenceError) is a
        # genuine failure — already engraved by glyff when it escaped the step.
        # Retrying would only waste the budget, so it propagates immediately.
        middleware = Retrier(max_retries=3)
        calls = 0
        original = RuntimeError("boom")

        async def failing():
            nonlocal calls
            calls += 1
            raise original

        with pytest.raises(RuntimeError) as exc_info:
            await middleware.wrap(_ctx(), failing)
        assert exc_info.value is original
        assert calls == 1

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
                raise InvalidInferenceResponseError("bad response")
            return "ok"

        assert await middleware.wrap(_ctx(), fail_then_succeed) == "ok"
        assert calls == 2

    async def test_inference_error_yields_after_exhausting_retries(self):
        # When the recoverable budget is spent, the original InferenceError is
        # re-raised untouched. It is a PauseException, so it propagates as a
        # non-engraved, resumable interrupt rather than a hard, engraved failure.
        middleware = Retrier(max_retries=2)
        calls = 0
        original = InvalidInferenceResponseError("still invalid")

        async def always_fail():
            nonlocal calls
            calls += 1
            raise original

        with pytest.raises(InvalidInferenceResponseError) as exc_info:
            await middleware.wrap(_ctx(), always_fail)

        assert exc_info.value is original
        assert isinstance(exc_info.value, PauseException)
        # initial attempt + 2 retries
        assert calls == 3

    async def test_does_not_retry_yield_exception(self):
        middleware = Retrier(max_retries=3)

        async def interrupt():
            raise PauseException("resume later")

        with pytest.raises(PauseException):
            await middleware.wrap(_ctx(), interrupt)
