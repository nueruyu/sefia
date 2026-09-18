from abc import ABC, abstractmethod
from typing import Any


class SessionStorage(ABC):
    """
    Abstract interface for persisting session-scoped key-value data.

    Implementations back the session state facility (``StateStore`` /
    ``get_state``). Writes are expected to commit immediately, so state written
    before a pause survives the interrupt and is visible when the run resumes.

    ``set_if_absent`` is the atomic primitive used by durable external
    integrations to accept one immutable value without a read/write race.
    ``keys`` provides prefix discovery for execution-owned resources such as
    pending external actions.
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
    async def set_if_absent(self, key: str, value: Any, type_hint: type) -> bool:
        """Atomically set ``key`` only when it does not already exist.

        Returns ``True`` when this call stored the value and ``False`` when
        another value was already present.
        """
        ...

    @abstractmethod
    async def keys(self, prefix: str = "") -> tuple[str, ...]:
        """Returns persisted keys beginning with ``prefix``, sorted."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Deletes a key."""
        ...
