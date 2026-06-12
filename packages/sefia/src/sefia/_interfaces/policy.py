from abc import ABC

from .event_handler import EventHandler
from .middleware import InferenceMiddleware, StepMiddleware


class Policy(ABC):
    """
    Abstract base class for a policy that can be applied to an @infer call.

    A policy contributes observation handlers and/or middleware to an inference run.
    """

    def create_handlers(self) -> list[EventHandler]:
        """Create observation handlers used by this policy (default: none)."""
        return []

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        """Create middleware used by this policy (default: none)."""
        return []
