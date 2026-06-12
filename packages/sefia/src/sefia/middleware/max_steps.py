from typing import Awaitable, Callable

from .._interfaces.middleware import StepContext, StepMiddleware
from ..exceptions import InferenceControlSignal
from ..inference import InferenceDecision


class MaxStepsExceededError(InferenceControlSignal):
    pass


class StepLimiter(StepMiddleware):
    def __init__(self, max_steps: int | None):
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be at least 1 or None")
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
