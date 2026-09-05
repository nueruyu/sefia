"""Reusable pytest contract for ``SessionStorage`` implementations."""

from collections.abc import Callable
from typing import TypeAlias

from pydantic import BaseModel

from ..storage import SessionStorage

SessionStorageFactory: TypeAlias = Callable[[], SessionStorage]


class _StoredValue(BaseModel):
    value: str


class SessionStorageContract:
    """Shared key-value behavior required by session storage implementations."""

    async def test_missing_key_returns_none(
        self, session_storage_factory: SessionStorageFactory
    ) -> None:
        assert await session_storage_factory().get("missing", dict) is None

    async def test_value_round_trips_after_reopening(
        self, session_storage_factory: SessionStorageFactory
    ) -> None:
        value = _StoredValue(value="kept")
        await session_storage_factory().set("state", value, _StoredValue)

        restored = await session_storage_factory().get("state", _StoredValue)

        assert restored == value
        assert isinstance(restored, _StoredValue)

    async def test_overwrite_replaces_the_value(
        self, session_storage_factory: SessionStorageFactory
    ) -> None:
        storage = session_storage_factory()
        await storage.set("state", {"value": "first"}, dict)

        await storage.set("state", {"value": "second"}, dict)

        assert await session_storage_factory().get("state", dict) == {"value": "second"}

    async def test_delete_removes_only_the_selected_key(
        self, session_storage_factory: SessionStorageFactory
    ) -> None:
        storage = session_storage_factory()
        await storage.set("first", {"value": 1}, dict)
        await storage.set("second", {"value": 2}, dict)

        await storage.delete("first")

        reopened = session_storage_factory()
        assert await reopened.get("first", dict) is None
        assert await reopened.get("second", dict) == {"value": 2}


__all__ = ["SessionStorageContract", "SessionStorageFactory"]
