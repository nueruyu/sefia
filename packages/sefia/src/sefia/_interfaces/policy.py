from abc import ABC

from ..event_system import EventHandler
from .middleware import InferenceMiddleware, StepMiddleware


class Policy(ABC):
    """
    Abstract base class for a policy that can be applied to an @infer call.

    A policy contributes two kinds of extension to an inference run:

    - Observation via ``create_handlers``, which returns event handlers that are
      notified of events but cannot steer the loop.
    - Control via ``create_middleware``, which returns middleware that wraps the
      inference run or each step and can steer the executor's loops by retrying
      or raising exceptions.

    Both default to contributing nothing, so a policy only implements the kind it
    needs.
    """

    def create_handlers(self) -> list[EventHandler]:
        """Create observation handlers used by this policy (default: none)."""
        return []

    def create_middleware(self) -> list[InferenceMiddleware | StepMiddleware]:
        """Create control middleware used by this policy (default: none)."""
        return []
