from dataclasses import dataclass

from sefia._interfaces import InferenceMiddleware, Policy, StepMiddleware
from sefia.middleware.max_steps import StepLimiter


@dataclass
class MaxSteps(Policy):
    """A policy that caps the number of steps in a single inference loop."""

    count: int = 25

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        return [StepLimiter(max_steps=self.count)]
