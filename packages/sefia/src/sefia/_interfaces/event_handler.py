from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

from ..event_system import Event

E = TypeVar("E", bound=Event)


class EventHandler(ABC, Generic[E]):
    """
    Abstract base class for a handler that processes a specific type of event.
    """

    @property
    @abstractmethod
    def event_types(self) -> tuple[Type[Event], ...]:
        """Returns a tuple of event types that this handler can process."""
        ...

    @abstractmethod
    async def handle(self, event: E) -> None:
        """Process a specific inference event."""
        ...
