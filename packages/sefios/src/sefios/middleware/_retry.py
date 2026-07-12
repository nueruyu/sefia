from typing import Any, Awaitable, Callable

from sefia._interfaces.middleware import InferenceContext, InferenceMiddleware
from sefia.exceptions import InferenceError


class Retrier(InferenceMiddleware):
    """
    Retries a recoverable inference failure within the current process.

    Only a recoverable ``InferenceError`` (a transient provider hiccup, or an
    invalid LLM response) is retried: the inference run is restarted, up to
    ``max_retries`` times. Once the budget is spent, the *original* error is
    re-raised untouched. Because an ``InferenceError`` is also a
    ``PauseException``, that propagates as a graceful pause — the run stops and a
    later re-invocation can still recover the step (glyff leaves the interrupted
    execution in its ``STARTED`` state, so it re-runs on resume) rather than
    surfacing as a hard failure. In other words, retries here are an in-process
    fast path, with a durable resume as the fallback.

    Everything else propagates untouched: framework limits (max steps,
    stagnation), intentional ``PauseException`` interrupts, and any genuine,
    deterministic failure. Retrying those would only
    waste the budget and delay surfacing the real error. Tool failures never
    surface here either: the executor stringifies them into the history and
    feeds them back to the model, so the model can recover.

    Invalid LLM responses rarely reach this middleware in practice:
    ``LLMInferenceStrategy`` first retries them in place with corrective
    feedback (``max_repair_attempts``), which is the better repair path — it
    can tell the model what was wrong, while this middleware can only restart
    the run with an identical prompt. ``Retrier`` remains useful as the outer
    net for transient provider failures and for repair budgets that ran out.

    The retry counter is kept on the instance and persists across the attempts
    of a single run. Middleware is instantiated per inference run
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
                # The only retryable failure. On exhaustion, re-raise the
                # original error: it is a PauseException, so it propagates as a
                # resumable pause instead of a hard failure.
                # Any other exception is not caught here and propagates as-is.
                if self._retries_used >= self.max_retries:
                    raise
                self._retries_used += 1
