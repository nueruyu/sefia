from abc import ABC, abstractmethod
from typing import Any


class SessionStorage(ABC):
    """
    Abstract interface for persisting session-scoped key-value data.

    Implementations back the session state facility (``StateStore`` /
    ``get_state``). Writes are expected to commit immediately, so state written
    before a pause survives the interrupt and is visible when the run resumes.
    """

    @abstractmethod
    async def get(self, key: str, type_hint: type) -> Any | None:
        """Gets a value by its key."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, type_hint: type) -> None:
        """Sets a key-value pair."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Deletes a key."""
        ...

    @abstractmethod
    async def set_if_absent(self, key: str, value: Any, type_hint: type) -> bool:
        """Insert atomically across bindings to the same logical store.

        Return True if inserted; False leaves an existing value unchanged.
        """
        ...

    @abstractmethod
    async def keys(self, prefix: str) -> list[str]:
        """Return matching keys in lexicographic order (not a snapshot)."""
        ...
