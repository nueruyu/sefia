from typing import Any, Awaitable, Callable

from glyff.exceptions import YieldException

from sefia._interfaces.middleware import InferenceContext, InferenceMiddleware
from sefia.exceptions import InferenceError, SefiaError


class MaxRetriesExceededError(SefiaError):
    """Raised when the configured number of retries has been exhausted."""


class Retrier(InferenceMiddleware):
    """
    Restarts the inference run when an inference attempt fails.

    Two kinds of failure are retried, with different exhaustion behavior:

    - A recoverable ``InferenceError`` (a transient provider hiccup, or an
      invalid LLM response) is retried within this process. Once the budget is
      spent, the *original* error is re-raised untouched. Because an
      ``InferenceError`` is also a ``YieldException``, that propagates as a
      graceful, non-engraved interrupt: the run pauses and a later re-invocation
      can still recover the step, rather than the failure being engraved as a
      permanent ``FAILED`` record. In other words, retries here are an in-process
      fast path, with a durable resume as the fallback.
    - Any other exception is retried too, but once the budget is spent it is
      wrapped in ``MaxRetriesExceededError`` (a ``SefiaError``), which is a
      genuine, engraved failure.

    Framework exceptions (for example max steps, stagnation, or an
    already-exhausted retry budget) and intentional ``YieldException`` interrupts
    are allowed to propagate untouched, so retries are never wasted on a
    deterministic limit. Tool failures are not retried either: the executor
    stringifies them into the history and feeds them back to the model, so they
    never surface here as exceptions.

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
            except InferenceError:
                # Recoverable. ``InferenceError`` is checked before the
                # ``(SefiaError, YieldException)`` branch because it subclasses
                # both. On exhaustion, re-raise the original error: it is a
                # YieldException, so it propagates as a non-engraved, resumable
                # interrupt instead of a hard failure.
                if self._retries_used >= self.max_retries:
                    raise
                self._retries_used += 1
            except (SefiaError, YieldException):
                raise
            except Exception as e:
                if self._retries_used >= self.max_retries:
                    raise MaxRetriesExceededError(
                        f"Failed after {self.max_retries} retries."
                    ) from e
                self._retries_used += 1
