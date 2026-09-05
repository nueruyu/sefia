from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypeAlias, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import sefios.storage as storage_module
from glyff import Serializer
from glyff_pydantic import PydanticSerializer
from pytest_mock import MockerFixture

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


async def test_file_store_commits_immediately(tmp_path: Path) -> None:
    serializer = PydanticSerializer()
    store = FileSessionStorage(base_dir=tmp_path, serializer=serializer)

    await store.set("session/state", {"value": "kept"}, dict)

    # A separately constructed store over the same directory observes the
    # write, confirming set() commits to disk immediately rather than staging.
    reader = FileSessionStorage(base_dir=tmp_path, serializer=serializer)
    assert await reader.get("session/state", dict) == {"value": "kept"}


async def test_file_store_keys_with_dotted_tails_do_not_collide(tmp_path: Path) -> None:
    store = FileSessionStorage(base_dir=tmp_path, serializer=PydanticSerializer())

    await store.set("state.v1", {"value": "one"}, dict)
    await store.set("state.v2", {"value": "two"}, dict)

    assert await store.get("state.v1", dict) == {"value": "one"}
    assert await store.get("state.v2", dict) == {"value": "two"}


async def test_file_store_rejects_unsafe_key_parts(tmp_path: Path) -> None:
    store = FileSessionStorage(base_dir=tmp_path, serializer=PydanticSerializer())
    with pytest.raises(ValueError):
        await store.get("..", dict)


async def test_sqlite_store_is_visible_to_a_new_instance(tmp_path: Path) -> None:
    database = tmp_path / "sessions.sqlite3"
    serializer = PydanticSerializer()
    writer = SQLiteSessionStorage(database, "session", serializer)

    await writer.set("state", {"value": "kept"}, dict)

    reader = SQLiteSessionStorage(database, "session", serializer)
    assert await reader.get("state", dict) == {"value": "kept"}


async def test_sqlite_store_isolates_sessions(tmp_path: Path) -> None:
    database = tmp_path / "sessions.sqlite3"
    serializer = PydanticSerializer()
    first = SQLiteSessionStorage(database, "first", serializer)
    second = SQLiteSessionStorage(database, "second", serializer)

    await first.set("state", {"value": "first"}, dict)
    await second.set("state", {"value": "second"}, dict)

    assert await first.get("state", dict) == {"value": "first"}
    assert await second.get("state", dict) == {"value": "second"}


async def test_sqlite_store_closes_every_connection(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    connection = MagicMock(spec=sqlite3.Connection)
    connection.execute.return_value.fetchone.return_value = None
    mocker.patch.object(SQLiteSessionStorage, "_connect", return_value=connection)
    store = SQLiteSessionStorage(
        tmp_path / "sessions.sqlite3", "session", PydanticSerializer()
    )

    await store.get("state", dict)
    await store.set("state", {"value": "kept"}, dict)
    await store.delete("state")

    assert connection.close.call_count == 4
