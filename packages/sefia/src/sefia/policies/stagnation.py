from dataclasses import dataclass

from sefia import InferenceMiddleware, Policy, StepMiddleware
from sefia.middleware.stagnation import StagnationDetector


@dataclass
class StagnationPolicy(Policy):
    max_repeats: int = 3

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        return [StagnationDetector(max_repeats=self.max_repeats)]
