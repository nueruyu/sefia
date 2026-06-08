from dataclasses import dataclass

from sefia.interfaces import InferenceMiddleware, Policy, StepMiddleware
from sefia.middleware.retry import RetryMiddleware


@dataclass
class MaxRetries(Policy):
    """A policy that specifies the maximum number of retries on failure."""

    count: int

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        return [RetryMiddleware(max_retries=self.count)]
