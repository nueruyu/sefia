"""Shared behavior required of every session storage."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias, cast
from unittest.mock import AsyncMock

import pytest
import sefios.storage as storage_module
from glyff import Serializer
from glyff_pydantic import PydanticSerializer

from sefios.storage import (
    FileSessionStorage,
    MemorySessionStorage,
    SessionStorage,
    SQLiteSessionStorage,
)


StoreType: TypeAlias = (
    type[FileSessionStorage] | type[MemorySessionStorage] | type[SQLiteSessionStorage]
)
STORE_TYPES: tuple[StoreType, ...] = (
    FileSessionStorage,
    MemorySessionStorage,
    SQLiteSessionStorage,
)


def _make_store(
    store_type: StoreType, tmp_path: Path, serializer: Serializer
) -> SessionStorage:
    if store_type is FileSessionStorage:
        return store_type(base_dir=tmp_path, serializer=serializer)
    if store_type is SQLiteSessionStorage:
        return store_type(tmp_path / "state.sqlite3", "session", serializer)
    return store_type(serializer=serializer)


@pytest.mark.parametrize("store_type", STORE_TYPES)
async def test_get_awaits_deserialize(store_type: StoreType, tmp_path: Path) -> None:
    serializer = AsyncMock()
    serializer.serialize.return_value = b'{"value": "loaded"}'
    serializer.deserialize.return_value = {"value": "loaded"}

    store = _make_store(store_type, tmp_path, cast(Serializer, serializer))

    # Seed a value so it is persisted in the store's own backing, then read it
    # back — get() must deserialize the stored bytes.
    await store.set("session/state", {"value": "loaded"}, dict)
    result = await store.get("session/state", dict)

    assert result == {"value": "loaded"}
    serializer.deserialize.assert_awaited_once_with(b'{"value": "loaded"}', dict)


@pytest.mark.parametrize("store_type", STORE_TYPES)
async def test_get_returns_none_when_data_is_missing(
    store_type: StoreType, tmp_path: Path
) -> None:
    serializer = AsyncMock()
    store = _make_store(store_type, tmp_path, cast(Serializer, serializer))

    result = await store.get("session/state", dict)

    assert result is None
    serializer.deserialize.assert_not_awaited()


@pytest.mark.parametrize("store_type", STORE_TYPES)
async def test_set_then_delete_round_trip(
    store_type: StoreType, tmp_path: Path
) -> None:
    serializer = PydanticSerializer()
    store = _make_store(store_type, tmp_path, serializer)

    await store.set("session/state", {"value": "kept"}, dict)
    assert await store.get("session/state", dict) == {"value": "kept"}

    await store.delete("session/state")
    assert await store.get("session/state", dict) is None


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in storage_module.__all__
        if isinstance(value := getattr(storage_module, name), type)
        and value is not SessionStorage
        and issubclass(value, SessionStorage)
    }

    assert set(STORE_TYPES) == exported


@pytest.mark.parametrize("store_type", STORE_TYPES)
async def test_overwrite_replaces_the_stored_value(
    store_type: StoreType, tmp_path: Path, serializer: Serializer
) -> None:
    store = _make_store(store_type, tmp_path, serializer)
    await store.set("session/state", {"value": "first"}, dict)

    await store.set("session/state", {"value": "second"}, dict)

    assert await store.get("session/state", dict) == {"value": "second"}
