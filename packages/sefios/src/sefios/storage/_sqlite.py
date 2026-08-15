import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from glyff import Serializer
from typing_extensions import final, override

from ._base import SessionStorage


@final
class SQLiteSessionStorage(SessionStorage):
    """SQLite-backed storage for one session's state."""

    def __init__(
        self, database: str | Path, session_id: str, serializer: Serializer
    ) -> None:
        self._database = Path(database)
        self._session_id = session_id
        self._serializer = serializer
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database)

    def _initialize(self) -> None:
        self._database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sefia_session_state (
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value BLOB NOT NULL,
                    PRIMARY KEY (session_id, key)
                )
                """
            )

    def _read(self, key: str) -> bytes | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT value FROM sefia_session_state
                WHERE session_id = ? AND key = ?
                """,
                (self._session_id, key),
            ).fetchone()
        return bytes(row[0]) if row is not None else None

    def _write(self, key: str, value: bytes) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sefia_session_state (session_id, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT (session_id, key) DO UPDATE SET value = excluded.value
                """,
                (self._session_id, key, value),
            )

    def _delete(self, key: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM sefia_session_state
                WHERE session_id = ? AND key = ?
                """,
                (self._session_id, key),
            )

    @override
    async def get(self, key: str, type_hint: type) -> Any | None:
        value = await asyncio.to_thread(self._read, key)
        if value is None:
            return None
        return await self._serializer.deserialize(value, type_hint)

    @override
    async def set(self, key: str, value: Any, type_hint: type) -> None:
        serialized = await self._serializer.serialize(value, type_hint)
        await asyncio.to_thread(self._write, key, serialized)

    @override
    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._delete, key)
