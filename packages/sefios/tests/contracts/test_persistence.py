"""Apply the public persistence contract to every built-in provider."""

import inspect
from pathlib import Path

import pytest

import sefios
from sefios import (
    FilePersistence,
    MemoryPersistence,
    PersistenceProvider,
    SQLitePersistence,
)
from sefios.testing import PersistenceProviderContract

PROVIDER_TYPES = (FilePersistence, MemoryPersistence, SQLitePersistence)


class TestMemoryPersistenceContract(PersistenceProviderContract):
    @pytest.fixture
    def persistence_provider(self) -> PersistenceProvider:
        return MemoryPersistence()


class TestFilePersistenceContract(PersistenceProviderContract):
    @pytest.fixture
    def persistence_provider(self, tmp_path: Path) -> PersistenceProvider:
        return FilePersistence(tmp_path / "sessions")


class TestSQLitePersistenceContract(PersistenceProviderContract):
    @pytest.fixture
    def persistence_provider(self, tmp_path: Path) -> PersistenceProvider:
        return SQLitePersistence(tmp_path / "sessions.sqlite3")


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in sefios.__all__
        if inspect.isclass(value := getattr(sefios, name))
        and value is not PersistenceProvider
        and issubclass(value, PersistenceProvider)
    }

    assert set(PROVIDER_TYPES) == exported
