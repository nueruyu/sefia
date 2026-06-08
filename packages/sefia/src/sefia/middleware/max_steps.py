from typing import Awaitable, Callable

from ..interfaces.middleware import StepContext, StepMiddleware
from ..models import InferenceDecision
from .signals import MaxStepsExceededError


class MaxStepsMiddleware(StepMiddleware):
    """
    Stops the inference loop once it would exceed a maximum number of steps.

    ``StepContext.step`` is a 0-based index; this middleware refuses to start a
    step once that index reaches ``max_steps`` (i.e. after ``max_steps`` steps
    have already run), raising ``MaxStepsExceededError``. Pass ``None`` for no
    limit.
    """

    def __init__(self, max_steps: int | None):
        self.max_steps = max_steps

    async def wrap(
        self,
        ctx: StepContext,
        nxt: Callable[[], Awaitable[InferenceDecision]],
    ) -> InferenceDecision:
        if self.max_steps is not None and ctx.step >= self.max_steps:
            raise MaxStepsExceededError(
                f"Inference exceeded the maximum of {self.max_steps} step(s)."
            )
        return await nxt()
