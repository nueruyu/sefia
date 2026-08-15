from pathlib import Path
from typing import Protocol, final

from glyff import Backend
from glyff.store import MemoryBackend
from glyff_pydantic import PydanticSerializer
from glyff_sqlite import SQLiteBackend

from .storage import (
    FileSessionStorage,
    MemorySessionStorage,
    SessionStorage,
    SQLiteSessionStorage,
)


class PersistenceProvider(Protocol):
    """Creates the execution and state stores for one session."""

    def create_execution_backend(self) -> Backend: ...

    def create_session_storage(self, session_id: str) -> SessionStorage: ...


@final
class SQLitePersistenceProvider:
    """Durable local persistence backed by one SQLite database."""

    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)

    def create_execution_backend(self) -> Backend:
        return SQLiteBackend(self.database)

    def create_session_storage(self, session_id: str) -> SessionStorage:
        return SQLiteSessionStorage(self.database, session_id, PydanticSerializer())


@final
class MemoryPersistenceProvider:
    """Ephemeral persistence shared for the lifetime of this object."""

    def __init__(self) -> None:
        self._backend = MemoryBackend()
        self._session_storages: dict[str, MemorySessionStorage] = {}

    def create_execution_backend(self) -> Backend:
        return self._backend

    def create_session_storage(self, session_id: str) -> SessionStorage:
        storage = self._session_storages.get(session_id)
        if storage is None:
            storage = MemorySessionStorage(PydanticSerializer())
            self._session_storages[session_id] = storage
        return storage


@final
class FilePersistenceProvider:
    """Debug-oriented JSON file persistence provided by ``sefios[file-store]``."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)

    def create_execution_backend(self) -> Backend:
        try:
            from glyff_file_store import JsonFileBackend
        except ImportError as error:
            raise ImportError(
                "The 'file-store' extra is required for FilePersistenceProvider. "
                "Install it with: pip install 'sefios[file-store]'"
            ) from error
        return JsonFileBackend(base_dir=self.session_dir / "glyff_sessions")

    def create_session_storage(self, session_id: str) -> SessionStorage:
        return FileSessionStorage(
            base_dir=self.session_dir / "sefia_metadata" / session_id,
            serializer=PydanticSerializer(),
        )
