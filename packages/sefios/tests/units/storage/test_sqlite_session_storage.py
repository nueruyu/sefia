import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

from glyff_pydantic import PydanticSerializer
from pytest_mock import MockerFixture

from sefios.storage import SQLiteSessionStorage


async def test_is_visible_to_a_new_instance(tmp_path: Path) -> None:
    database = tmp_path / "sessions.sqlite3"
    serializer = PydanticSerializer()
    writer = SQLiteSessionStorage(database, "session", serializer)

    await writer.set("state", {"value": "kept"}, dict)

    reader = SQLiteSessionStorage(database, "session", serializer)
    assert await reader.get("state", dict) == {"value": "kept"}


async def test_isolates_sessions(tmp_path: Path) -> None:
    database = tmp_path / "sessions.sqlite3"
    serializer = PydanticSerializer()
    first = SQLiteSessionStorage(database, "first", serializer)
    second = SQLiteSessionStorage(database, "second", serializer)

    await first.set("state", {"value": "first"}, dict)
    await second.set("state", {"value": "second"}, dict)

    assert await first.get("state", dict) == {"value": "first"}
    assert await second.get("state", dict) == {"value": "second"}


async def test_closes_every_connection(tmp_path: Path, mocker: MockerFixture) -> None:
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
