"""Apply the public session-registry contract to every built-in registry."""

from pathlib import Path

import pytest
import sefios.sessions as sessions
from sefios.sessions import (
    FileSessionRegistry,
    MemorySessionRegistry,
    SessionRegistry,
    SQLiteSessionRegistry,
)
from sefios.testing import SessionRegistryContract, SessionRegistryFactory

REGISTRY_TYPES = (FileSessionRegistry, MemorySessionRegistry, SQLiteSessionRegistry)


class TestMemorySessionRegistryContract(SessionRegistryContract):
    @pytest.fixture
    def session_registry_factory(self) -> SessionRegistryFactory:
        registry = MemorySessionRegistry()
        return lambda: registry


class TestFileSessionRegistryContract(SessionRegistryContract):
    @pytest.fixture
    def session_registry_factory(self, tmp_path: Path) -> SessionRegistryFactory:
        return lambda: FileSessionRegistry(tmp_path / "sessions.txt")


class TestSQLiteSessionRegistryContract(SessionRegistryContract):
    @pytest.fixture
    def session_registry_factory(self, tmp_path: Path) -> SessionRegistryFactory:
        return lambda: SQLiteSessionRegistry(tmp_path / "sessions.sqlite3")


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in sessions.__all__
        if isinstance(value := getattr(sessions, name), type)
        and value is not SessionRegistry
        and issubclass(value, SessionRegistry)
    }

    assert set(REGISTRY_TYPES) == exported
