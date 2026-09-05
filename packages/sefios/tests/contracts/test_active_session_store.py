"""Apply the public active-session contract to every built-in store."""

import inspect
from pathlib import Path

import pytest
import sefios.sessions as sessions
from sefios.sessions import (
    ActiveSessionStore,
    FileActiveSessionStore,
    MemoryActiveSessionStore,
)
from sefios.testing import ActiveSessionStoreContract, ActiveSessionStoreFactory

ACTIVE_STORE_TYPES = (FileActiveSessionStore, MemoryActiveSessionStore)


class TestMemoryActiveSessionStoreContract(ActiveSessionStoreContract):
    @pytest.fixture
    def active_session_store_factory(self) -> ActiveSessionStoreFactory:
        store = MemoryActiveSessionStore()
        return lambda: store


class TestFileActiveSessionStoreContract(ActiveSessionStoreContract):
    @pytest.fixture
    def active_session_store_factory(self, tmp_path: Path) -> ActiveSessionStoreFactory:
        return lambda: FileActiveSessionStore(tmp_path / "active-session.txt")


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in sessions.__all__
        if inspect.isclass(value := getattr(sessions, name))
        and value is not ActiveSessionStore
        and issubclass(value, ActiveSessionStore)
    }

    assert set(ACTIVE_STORE_TYPES) == exported
