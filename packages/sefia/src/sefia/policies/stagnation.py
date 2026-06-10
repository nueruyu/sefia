from dataclasses import dataclass

from sefia.interfaces import InferenceMiddleware, Policy, StepMiddleware
from sefia.middleware.stagnation import StagnationDetector


@dataclass
class StagnationPolicy(Policy):
    """
    A policy that adds middleware to detect and prevent infinite loops
    of the same tool call.
    """

    max_repeats: int = 3

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        """Creates a new StagnationDetector instance."""
        return [StagnationDetector(max_repeats=self.max_repeats)]
