"""Shared behavior required of every active-session store."""

import inspect
from pathlib import Path
from typing import TypeAlias

import pytest

import sefios.sessions as sessions
from sefios.sessions import (
    ActiveSessionStore,
    FileActiveSessionStore,
    MemoryActiveSessionStore,
)


ActiveStoreType: TypeAlias = (
    type[FileActiveSessionStore] | type[MemoryActiveSessionStore]
)
ACTIVE_STORE_TYPES: tuple[ActiveStoreType, ...] = (
    FileActiveSessionStore,
    MemoryActiveSessionStore,
)


@pytest.fixture(params=ACTIVE_STORE_TYPES, ids=lambda store_type: store_type.__name__)
def active_store(request: pytest.FixtureRequest, tmp_path: Path) -> ActiveSessionStore:
    store_type = request.param
    if store_type is FileActiveSessionStore:
        return store_type(tmp_path / "active-session.txt")
    return store_type()


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in sessions.__all__
        if inspect.isclass(value := getattr(sessions, name))
        and value is not ActiveSessionStore
        and issubclass(value, ActiveSessionStore)
    }

    assert set(ACTIVE_STORE_TYPES) == exported


def test_contract_is_empty_initially(active_store: ActiveSessionStore) -> None:
    assert active_store.get_active_session_id() is None


def test_contract_stores_and_replaces_the_active_session(
    active_store: ActiveSessionStore,
) -> None:
    active_store.set_active_session_id("first")
    assert active_store.get_active_session_id() == "first"

    active_store.set_active_session_id("second")
    assert active_store.get_active_session_id() == "second"
