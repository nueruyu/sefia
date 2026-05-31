from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StateStore(ABC, Generic[T]):
    """
    Abstract interface for managing a specific piece of state.
    """

    @abstractmethod
    async def ensure(self) -> T:
        """
        Ensures the state is loaded. If it doesn't exist, returns a
        default-initialized instance.
        """
        ...

    @abstractmethod
    async def get(self, default: T | None = None) -> T | None:
        """
        Returns the state if it exists, otherwise default (None if not specified).
        """
        ...

    @abstractmethod
    async def save(self, state: T) -> None:
        """Saves the new state."""
        ...

    @abstractmethod
    async def delete(self) -> None:
        """Deletes the state."""
        ...
