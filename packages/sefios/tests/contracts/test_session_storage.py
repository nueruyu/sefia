"""Apply the public session-storage contract to every built-in store."""

from pathlib import Path

import pytest
import sefios.storage as storage_module
from glyff import Serializer

from sefios.storage import (
    FileSessionStorage,
    MemorySessionStorage,
    SessionStorage,
    SQLiteSessionStorage,
)
from sefios.testing import SessionStorageContract, SessionStorageFactory

STORE_TYPES = (FileSessionStorage, MemorySessionStorage, SQLiteSessionStorage)


class TestMemorySessionStorageContract(SessionStorageContract):
    @pytest.fixture
    def session_storage_factory(self, serializer: Serializer) -> SessionStorageFactory:
        storage = MemorySessionStorage(serializer)
        return lambda: storage


class TestFileSessionStorageContract(SessionStorageContract):
    @pytest.fixture
    def session_storage_factory(
        self, tmp_path: Path, serializer: Serializer
    ) -> SessionStorageFactory:
        return lambda: FileSessionStorage(tmp_path, serializer)


class TestSQLiteSessionStorageContract(SessionStorageContract):
    @pytest.fixture
    def session_storage_factory(
        self, tmp_path: Path, serializer: Serializer
    ) -> SessionStorageFactory:
        return lambda: SQLiteSessionStorage(
            tmp_path / "state.sqlite3", "session", serializer
        )


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in storage_module.__all__
        if isinstance(value := getattr(storage_module, name), type)
        and value is not SessionStorage
        and issubclass(value, SessionStorage)
    }

    assert set(STORE_TYPES) == exported
