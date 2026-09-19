"""Reusable pytest contract for ``SessionStorage`` implementations."""

import asyncio
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

    async def test_conditional_insert_survives_reopening(self) -> None:
        assert await self.make_session_storage().set_if_absent("a/b", "first", str)
        assert not await self.make_session_storage().set_if_absent("a/b", "second", str)
        assert await self.make_session_storage().get("a/b", str) == "first"

    async def test_concurrent_conditional_insert(self) -> None:
        stores = [self.make_session_storage() for _ in range(12)]
        inserted = await asyncio.gather(
            *(store.set_if_absent("race", i, int) for i, store in enumerate(stores))
        )
        assert sum(inserted) == 1
        assert await self.make_session_storage().get("race", int) == inserted.index(
            True
        )

    async def test_prefix_keys_survive_reopening(self) -> None:
        keys = ["a/z", "a/b/c", "a/b", "other", "a/%?:日本語", "a.v1"]
        for key in keys:
            await self.make_session_storage().set(key, key, str)
        reopened = self.make_session_storage()
        assert await reopened.keys("a/") == sorted(keys[:3] + [keys[4]])
        assert await reopened.keys("") == sorted(keys)
        assert await reopened.keys("a/%") == [keys[4]]
        assert await reopened.keys("missing") == []


__all__ = ["SessionStorageContract", "SessionStorageFactory"]
