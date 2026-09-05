import inspect
from pathlib import Path
from typing import TypeAlias

import pytest
from glyff import Backend

import sefios
from sefios import (
    FilePersistence,
    MemoryPersistence,
    PersistenceProvider,
    SQLitePersistence,
)
from sefios.sessions import SessionRegistry
from sefios.storage import SessionStorage


ProviderType: TypeAlias = (
    type[FilePersistence] | type[MemoryPersistence] | type[SQLitePersistence]
)
PROVIDER_TYPES: tuple[ProviderType, ...] = (
    FilePersistence,
    MemoryPersistence,
    SQLitePersistence,
)


@pytest.fixture(params=PROVIDER_TYPES, ids=lambda provider_type: provider_type.__name__)
def persistence(request: pytest.FixtureRequest, tmp_path: Path) -> PersistenceProvider:
    provider_type = request.param
    if provider_type is FilePersistence:
        return provider_type(tmp_path / "sessions")
    if provider_type is SQLitePersistence:
        return provider_type(tmp_path / "sessions.sqlite3")
    return provider_type()


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in sefios.__all__
        if inspect.isclass(value := getattr(sefios, name))
        and value is not PersistenceProvider
        and issubclass(value, PersistenceProvider)
    }

    assert set(PROVIDER_TYPES) == exported


def test_contract_creates_each_persistence_component(
    persistence: PersistenceProvider,
) -> None:
    assert isinstance(persistence.create_execution_backend(), Backend)
    assert isinstance(persistence.create_session_storage("session"), SessionStorage)
    assert isinstance(persistence.create_session_registry(), SessionRegistry)


async def test_contract_reuses_session_data_for_the_same_provider(
    persistence: PersistenceProvider,
) -> None:
    writer = persistence.create_session_storage("first")
    await writer.set("state", {"value": "kept"}, dict)

    assert await persistence.create_session_storage("first").get("state", dict) == {
        "value": "kept"
    }
    assert await persistence.create_session_storage("second").get("state", dict) is None


def test_contract_reuses_the_session_registry(
    persistence: PersistenceProvider,
) -> None:
    persistence.create_session_registry().register_session("session")

    assert persistence.create_session_registry().session_exists("session")
