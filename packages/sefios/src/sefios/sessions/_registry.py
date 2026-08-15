import sqlite3
import uuid
from abc import ABC, abstractmethod
from contextlib import closing
from pathlib import Path
from threading import Lock
from typing import final

from typing_extensions import override


class SessionRegistry(ABC):
    """Tracks the sessions available through a persistence provider."""

    @abstractmethod
    def session_exists(self, session_id: str) -> bool: ...

    @abstractmethod
    def register_session(self, session_id: str) -> None: ...

    def create_session(self) -> str:
        while True:
            session_id = str(uuid.uuid4())
            if not self.session_exists(session_id):
                self.register_session(session_id)
                return session_id


@final
class MemorySessionRegistry(SessionRegistry):
    def __init__(self) -> None:
        self._session_ids: set[str] = set()
        self._lock = Lock()

    @override
    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._session_ids

    @override
    def register_session(self, session_id: str) -> None:
        with self._lock:
            self._session_ids.add(session_id)


@final
class SQLiteSessionRegistry(SessionRegistry):
    def __init__(self, database: str | Path) -> None:
        self._database = Path(database)
        self._database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._database)) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sefia_sessions (
                    session_id TEXT PRIMARY KEY
                )
                """
            )

    @override
    def session_exists(self, session_id: str) -> bool:
        with closing(sqlite3.connect(self._database)) as connection:
            row = connection.execute(
                "SELECT 1 FROM sefia_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row is not None

    @override
    def register_session(self, session_id: str) -> None:
        with closing(sqlite3.connect(self._database)) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO sefia_sessions (session_id) VALUES (?)",
                (session_id,),
            )


@final
class FileSessionRegistry(SessionRegistry):
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _read_session_ids(self) -> set[str]:
        try:
            content = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return set()
        return {line.strip() for line in content.splitlines() if line.strip()}

    @override
    def session_exists(self, session_id: str) -> bool:
        return session_id in self._read_session_ids()

    @override
    def register_session(self, session_id: str) -> None:
        session_ids = self._read_session_ids()
        if session_id in session_ids:
            return
        session_ids.add(session_id)
        self._path.write_text(
            "".join(f"{registered_id}\n" for registered_id in sorted(session_ids)),
            encoding="utf-8",
        )
