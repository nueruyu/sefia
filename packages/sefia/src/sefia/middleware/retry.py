from typing import Any, Awaitable, Callable

from glyff.exceptions import YieldException

from .._interfaces.middleware import InferenceMiddleware, InferenceContext
from ..exceptions import InferenceControlSignal, RequestInferenceRetry


class MaxRetriesExceededError(InferenceControlSignal):
    """Raised when the configured number of retries has been exhausted."""


class Retrier(InferenceMiddleware):
    """Retries an inference run up to the configured limit."""

    def __init__(self, max_retries: int = 3):
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.max_retries = max_retries
        self._retries_used = 0

    async def wrap(self, ctx: InferenceContext, nxt: Callable[[], Awaitable[Any]]) -> Any:
        try:
            return await nxt()
        except (InferenceControlSignal, YieldException):
            raise
        except Exception as e:
            if self._retries_used < self.max_retries:
                self._retries_used += 1
                raise RequestInferenceRetry() from e
            raise MaxRetriesExceededError(
                f"Failed after {self.max_retries} retries."
            ) from e
