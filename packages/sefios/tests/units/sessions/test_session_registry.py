from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import pytest
import sefios.sessions as sessions
from sefios.sessions import (
    FileSessionRegistry,
    MemorySessionRegistry,
    SessionRegistry,
    SQLiteSessionRegistry,
)


RegistryType: TypeAlias = (
    type[FileSessionRegistry]
    | type[MemorySessionRegistry]
    | type[SQLiteSessionRegistry]
)
REGISTRY_TYPES: tuple[RegistryType, ...] = (
    FileSessionRegistry,
    MemorySessionRegistry,
    SQLiteSessionRegistry,
)


@pytest.fixture(params=REGISTRY_TYPES, ids=lambda registry_type: registry_type.__name__)
def registry_factory(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Callable[[], SessionRegistry]:
    if request.param is MemorySessionRegistry:
        registry = MemorySessionRegistry()
        return lambda: registry
    if request.param is SQLiteSessionRegistry:
        return lambda: SQLiteSessionRegistry(tmp_path / "sessions.sqlite3")
    return lambda: FileSessionRegistry(tmp_path / "sessions.txt")


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in sessions.__all__
        if isinstance(value := getattr(sessions, name), type)
        and value is not SessionRegistry
        and issubclass(value, SessionRegistry)
    }

    assert set(REGISTRY_TYPES) == exported


def test_registers_sessions(
    registry_factory: Callable[[], SessionRegistry],
) -> None:
    registry = registry_factory()

    assert not registry.session_exists("session-1")

    registry.register_session("session-1")

    assert registry_factory().session_exists("session-1")


def test_creates_unique_registered_sessions(
    registry_factory: Callable[[], SessionRegistry],
) -> None:
    registry = registry_factory()

    first = registry.create_session()
    second = registry.create_session()

    assert first != second
    assert registry.session_exists(first)
    assert registry.session_exists(second)
