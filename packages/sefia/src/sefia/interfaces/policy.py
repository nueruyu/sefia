from abc import ABC, abstractmethod

from .event_handler import EventHandler


class Policy(ABC):
    """
    Abstract base class for a policy that can be applied to an @infer call.
    A policy generates one or more event handlers to enforce its rules.
    """

    @abstractmethod
    def create_handlers(self) -> list[EventHandler]:
        """Create a list of handlers that enforce this policy."""
        ...
