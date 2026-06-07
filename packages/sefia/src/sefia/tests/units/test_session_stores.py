from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from sefia.stores import FileSessionStore, MemorySessionStore


@pytest.mark.parametrize(
    ("store_type", "expected_read_key"),
    [
        (FileSessionStore, Path("sefia", "metadata", "session", "state.json")),
        (MemorySessionStore, "sefia::metadata::session/state"),
    ],
)
async def test_get_awaits_deserialize(store_type: type, expected_read_key):
    client = AsyncMock()
    client.read.return_value = b'{"value": "loaded"}'

    serializer = AsyncMock()
    serializer.deserialize.return_value = {"value": "loaded"}

    store = store_type(client=client, serializer=serializer)

    result = await store.get("session/state", dict)

    assert result == {"value": "loaded"}
    client.read.assert_awaited_once_with(expected_read_key)
    serializer.deserialize.assert_awaited_once_with(b'{"value": "loaded"}', dict)


@pytest.mark.parametrize(
    ("store_type", "missing_data"),
    [
        (FileSessionStore, b""),
        (MemorySessionStore, None),
    ],
)
async def test_get_returns_none_when_data_is_missing(store_type: type, missing_data):
    client = AsyncMock()
    client.read.return_value = missing_data

    serializer = AsyncMock()
    store = store_type(client=client, serializer=serializer)

    result = await store.get("session/state", dict)

    assert result is None
    serializer.deserialize.assert_not_awaited()
