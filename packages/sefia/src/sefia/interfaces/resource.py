from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Resource(ABC, Generic[T]):
    """
    Abstract base class for a lightweight reference to a potentially large object.
    This allows passing references into and out of @infer functions without
    cluttering the LLM's context with the full object data.
    """

    @abstractmethod
    async def get(self) -> T:
        """Retrieve the underlying value."""
        ...

    @abstractmethod
    async def set(self, value: T) -> None:
        """Update the underlying value."""
        ...
