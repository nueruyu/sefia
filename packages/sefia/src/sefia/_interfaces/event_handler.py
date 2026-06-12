from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

from ..event_system import Event

E = TypeVar("E", bound=Event)


class EventHandler(ABC, Generic[E]):
    """Base class for an observer that receives specific event types."""

    @property
    @abstractmethod
    def event_types(self) -> tuple[Type[Event], ...]:
        """Return the event classes accepted by this observer."""
        ...

    @abstractmethod
    async def handle(self, event: E) -> None:
        """Receive one event instance."""
        ...
