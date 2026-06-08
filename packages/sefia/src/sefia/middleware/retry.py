from typing import Any, Awaitable, Callable

from glyff.exceptions import YieldException

from ..interfaces.middleware import InferenceMiddleware, RunContext
from .signals import (
    InferenceControlSignal,
    MaxRetriesExceededError,
    RequestInferenceRetry,
)


class RetryMiddleware(InferenceMiddleware):
    """
    Restarts the inference run when an inference attempt fails.

    Only failures arising from the inference process itself are retried. Terminal
    control signals (max steps, stagnation, an already-exhausted retry budget) and
    graceful interrupts (``YieldException``) are allowed to propagate untouched,
    so retries are never wasted on a deterministic limit. Tool failures are not
    retried either: the executor stringifies them into the history and feeds them
    back to the model, so they never surface here as exceptions.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._retries_used = 0

    async def wrap(self, ctx: RunContext, nxt: Callable[[], Awaitable[Any]]) -> Any:
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
