from sefia.interfaces import InferenceMiddleware, Policy, StepMiddleware
from sefia.middleware.stagnation import StagnationMiddleware


class StagnationPolicy(Policy):
    """
    A policy that adds middleware to detect and prevent infinite loops
    of the same tool call.
    """

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        """Creates a new StagnationMiddleware instance."""
        return [StagnationMiddleware()]
