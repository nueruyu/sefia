from dataclasses import dataclass

from sefia import InferenceMiddleware, Policy, StepMiddleware
from sefia.middleware.retry import Retrier


@dataclass
class MaxRetries(Policy):
    """A policy that specifies the maximum number of retries on failure."""

    count: int

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        return [Retrier(max_retries=self.count)]
