"""Reusable pytest contract for ``SessionStorage`` implementations."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeAlias

from pydantic import BaseModel

from ..storage import SessionStorage

SessionStorageFactory: TypeAlias = Callable[[], SessionStorage]


class _StoredValue(BaseModel):
    value: str


class SessionStorageContract(ABC):
    """Shared key-value behavior required by session storage implementations."""

    @abstractmethod
    def make_session_storage(self) -> SessionStorage:
        """Reopen the same logical store, initially empty for each test."""
        ...

    async def test_missing_key_returns_none(self) -> None:
        assert await self.make_session_storage().get("missing", dict) is None

    async def test_value_round_trips_after_reopening(self) -> None:
        value = _StoredValue(value="kept")
        await self.make_session_storage().set("state", value, _StoredValue)

        restored = await self.make_session_storage().get("state", _StoredValue)

        assert restored == value
        assert isinstance(restored, _StoredValue)

    async def test_overwrite_replaces_the_value(self) -> None:
        storage = self.make_session_storage()
        await storage.set("state", {"value": "first"}, dict)

        await storage.set("state", {"value": "second"}, dict)

        assert await self.make_session_storage().get("state", dict) == {
            "value": "second"
        }

    async def test_delete_removes_only_the_selected_key(self) -> None:
        storage = self.make_session_storage()
        await storage.set("first", {"value": 1}, dict)
        await storage.set("second", {"value": 2}, dict)

        await storage.delete("first")

        reopened = self.make_session_storage()
        assert await reopened.get("first", dict) is None
        assert await reopened.get("second", dict) == {"value": 2}


__all__ = ["SessionStorageContract", "SessionStorageFactory"]
