from dataclasses import dataclass

from sefia._interfaces import InferenceMiddleware, Policy, StepMiddleware
from sefios.middleware._input import InputCallComposer
from sefios.middleware._max_steps import StepLimiter
from sefios.middleware._stagnation import StagnationDetector


@dataclass
class DefaultPolicy(Policy):
    """Sefios default policy: step limit, HITL composition, and stagnation detection."""

    max_steps: int | None = 25
    max_repeats: int = 3

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        middleware: list[InferenceMiddleware | StepMiddleware] = []
        if self.max_steps is not None:
            middleware.append(StepLimiter(max_steps=self.max_steps))
        middleware.append(StagnationDetector(max_repeats=self.max_repeats))
        # Runs innermost so StagnationDetector observes the decision that will execute.
        middleware.append(InputCallComposer())
        return middleware
