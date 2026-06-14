from dataclasses import dataclass

from sefia._interfaces import InferenceMiddleware, Policy, StepMiddleware
from sefia.middleware._retry import Retrier


@dataclass
class MaxRetries(Policy):
    """A policy that specifies the maximum number of retries on failure."""

    count: int

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        return [Retrier(max_retries=self.count)]
