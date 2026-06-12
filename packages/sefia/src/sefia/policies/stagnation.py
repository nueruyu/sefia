from dataclasses import dataclass

from sefia._interfaces import InferenceMiddleware, Policy, StepMiddleware
from sefia.middleware.stagnation import StagnationDetector


@dataclass
class StagnationPolicy(Policy):
    max_repeats: int = 3

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        return [StagnationDetector(max_repeats=self.max_repeats)]
