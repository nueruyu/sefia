from typing import Any, Awaitable, Callable

from glyff.exceptions import YieldException

from sefia._interfaces.middleware import InferenceContext, InferenceMiddleware
from sefia.exceptions import InferenceException, SefiaError


class MaxRetriesExceededError(SefiaError):
    """Raised when the configured number of retries has been exhausted."""


class Retrier(InferenceMiddleware):
    """
    Restarts the inference run when an inference attempt fails.

    Only failures arising from the inference process itself are retried.
    Framework exceptions (for example max steps, stagnation, or an
    already-exhausted retry budget) and graceful interrupts (``YieldException``)
    are allowed to propagate untouched, so retries are never wasted on a
    deterministic limit. Provider failures translated to ``InferenceException``
    are retried. Tool failures are not retried either: the executor stringifies
    them into the history and feeds them back to the model, so they never surface
    here as exceptions.

    The retry counter is intentionally kept on the instance and persists across
    the attempts of a single run. Middleware is instantiated per inference run
    (``Policy.create_middleware`` is called once per ``@infer`` invocation in
    ``decorators._run``), so an instance is never shared across concurrent runs;
    its state is scoped to a single run.
    """

    def __init__(self, max_retries: int = 3):
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.max_retries = max_retries
        self._retries_used = 0

    async def wrap(
        self, ctx: InferenceContext, nxt: Callable[[], Awaitable[Any]]
    ) -> Any:
        while True:
            try:
                return await nxt()
            except InferenceException as e:
                if self._retries_used >= self.max_retries:
                    raise MaxRetriesExceededError(
                        f"Failed after {self.max_retries} retries."
                    ) from e
                self._retries_used += 1
            except (SefiaError, YieldException):
                raise
            except Exception as e:
                if self._retries_used >= self.max_retries:
                    raise MaxRetriesExceededError(
                        f"Failed after {self.max_retries} retries."
                    ) from e
                self._retries_used += 1
