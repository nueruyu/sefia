from abc import abstractmethod
from typing import Any

from .._interfaces.session_store import SessionStore


class BufferedSessionStore(SessionStore):
    """Base session store that provides read-your-writes consistency.

    The underlying clients stage writes and only make them visible to reads once
    the surrounding transaction commits. This base buffers the writes and
    deletes issued during the current transaction so that later reads in the
    same transaction observe them immediately, instead of returning the
    last-committed value.

    Subclasses implement the client-specific persistence in :meth:`_read`,
    :meth:`_stage_write`, and :meth:`_stage_delete`.
    """

    def __init__(self) -> None:
        self._pending: dict[str, Any] = {}
        self._deleted: set[str] = set()

    async def get(self, key: str, type_hint: type) -> Any | None:
        if key in self._pending:
            return self._pending[key]
        if key in self._deleted:
            return None
        return await self._read(key, type_hint)

    async def set(self, key: str, value: Any, type_hint: type) -> None:
        await self._stage_write(key, value, type_hint)
        self._pending[key] = value
        self._deleted.discard(key)

    async def delete(self, key: str) -> None:
        await self._stage_delete(key)
        self._pending.pop(key, None)
        self._deleted.add(key)

    @abstractmethod
    async def _read(self, key: str, type_hint: type) -> Any | None:
        """Read the last-committed value for ``key`` from the backing client."""
        ...

    @abstractmethod
    async def _stage_write(self, key: str, value: Any, type_hint: type) -> None:
        """Stage a write of ``value`` for ``key`` on the backing client."""
        ...

    @abstractmethod
    async def _stage_delete(self, key: str) -> None:
        """Stage a delete of ``key`` on the backing client."""
        ...
