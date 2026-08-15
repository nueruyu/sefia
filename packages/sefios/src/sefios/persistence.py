from pathlib import Path
from typing import Protocol

from glyff import Backend, Serializer
from glyff.store import MemoryBackend
from glyff_sqlite import SQLiteBackend
from typing_extensions import final

from .storage import (
    FileSessionStorage,
    MemorySessionStorage,
    SessionStorage,
    SQLiteSessionStorage,
)


class SessionPersistence(Protocol):
    """Creates the execution and state stores for one session."""

    def create_backend(self, session_id: str, serializer: Serializer) -> Backend: ...

    def create_session_storage(
        self, session_id: str, serializer: Serializer
    ) -> SessionStorage: ...


@final
class SQLitePersistence:
    """Durable local persistence backed by one SQLite database."""

    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)

    def create_backend(self, session_id: str, serializer: Serializer) -> Backend:
        return SQLiteBackend(self.database)

    def create_session_storage(
        self, session_id: str, serializer: Serializer
    ) -> SessionStorage:
        return SQLiteSessionStorage(self.database, session_id, serializer)


@final
class MemoryPersistence:
    """Ephemeral persistence shared for the lifetime of this object."""

    def __init__(self) -> None:
        self._backend = MemoryBackend()
        self._session_storages: dict[str, MemorySessionStorage] = {}

    def create_backend(self, session_id: str, serializer: Serializer) -> Backend:
        return self._backend

    def create_session_storage(
        self, session_id: str, serializer: Serializer
    ) -> SessionStorage:
        storage = self._session_storages.get(session_id)
        if storage is None:
            storage = MemorySessionStorage(serializer)
            self._session_storages[session_id] = storage
        return storage


@final
class FilePersistence:
    """Debug-oriented JSON file persistence provided by ``sefios[file]``."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)

    def create_backend(self, session_id: str, serializer: Serializer) -> Backend:
        try:
            from glyff_file_store import JsonFileBackend
        except ImportError as error:
            raise ImportError(
                "The 'file' extra is required for FilePersistence. "
                "Install it with: pip install 'sefios[file]'"
            ) from error
        return JsonFileBackend(base_dir=self.session_dir / "glyff_sessions")

    def create_session_storage(
        self, session_id: str, serializer: Serializer
    ) -> SessionStorage:
        return FileSessionStorage(
            base_dir=self.session_dir / "sefia_metadata" / session_id,
            serializer=serializer,
        )
