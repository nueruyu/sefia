from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from glyff_pydantic import PydanticSerializer

from sefios.storage import FileSessionStorage, MemorySessionStorage


def _make_store(store_type: type, tmp_path, serializer):
    if store_type is FileSessionStorage:
        return store_type(base_dir=tmp_path, serializer=serializer)
    return store_type(serializer=serializer)


@pytest.mark.parametrize("store_type", [FileSessionStorage, MemorySessionStorage])
async def test_get_awaits_deserialize(store_type: type, tmp_path):
    serializer = AsyncMock()
    serializer.serialize.return_value = b'{"value": "loaded"}'
    serializer.deserialize.return_value = {"value": "loaded"}

    store = _make_store(store_type, tmp_path, serializer)

    # Seed a value so it is persisted in the store's own backing, then read it
    # back — get() must deserialize the stored bytes.
    await store.set("session/state", {"value": "loaded"}, dict)
    result = await store.get("session/state", dict)

    assert result == {"value": "loaded"}
    serializer.deserialize.assert_awaited_once_with(b'{"value": "loaded"}', dict)


@pytest.mark.parametrize("store_type", [FileSessionStorage, MemorySessionStorage])
async def test_get_returns_none_when_data_is_missing(store_type: type, tmp_path):
    serializer = AsyncMock()
    store = _make_store(store_type, tmp_path, serializer)

    result = await store.get("session/state", dict)

    assert result is None
    serializer.deserialize.assert_not_awaited()


@pytest.mark.parametrize("store_type", [FileSessionStorage, MemorySessionStorage])
async def test_set_then_delete_round_trip(store_type: type, tmp_path):
    serializer = PydanticSerializer()
    store = _make_store(store_type, tmp_path, serializer)

    await store.set("session/state", {"value": "kept"}, dict)
    assert await store.get("session/state", dict) == {"value": "kept"}

    await store.delete("session/state")
    assert await store.get("session/state", dict) is None


async def test_file_store_commits_immediately(tmp_path):
    serializer = PydanticSerializer()
    store = FileSessionStorage(base_dir=tmp_path, serializer=serializer)

    await store.set("session/state", {"value": "kept"}, dict)

    # A separately constructed store over the same directory observes the
    # write, confirming set() commits to disk immediately rather than staging.
    reader = FileSessionStorage(base_dir=tmp_path, serializer=serializer)
    assert await reader.get("session/state", dict) == {"value": "kept"}


async def test_file_store_keys_with_dotted_tails_do_not_collide(tmp_path):
    store = FileSessionStorage(base_dir=tmp_path, serializer=PydanticSerializer())

    await store.set("state.v1", {"value": "one"}, dict)
    await store.set("state.v2", {"value": "two"}, dict)

    assert await store.get("state.v1", dict) == {"value": "one"}
    assert await store.get("state.v2", dict) == {"value": "two"}


async def test_file_store_rejects_unsafe_key_parts(tmp_path):
    store = FileSessionStorage(base_dir=tmp_path, serializer=PydanticSerializer())
    with pytest.raises(ValueError):
        await store.get("..", dict)
