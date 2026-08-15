from collections.abc import Callable
from pathlib import Path

import pytest
from sefios.sessions import (
    FileSessionRegistry,
    MemorySessionRegistry,
    SessionRegistry,
    SQLiteSessionRegistry,
)


@pytest.fixture(params=["memory", "sqlite", "file"])
def registry_factory(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Callable[[], SessionRegistry]:
    if request.param == "memory":
        registry = MemorySessionRegistry()
        return lambda: registry
    if request.param == "sqlite":
        return lambda: SQLiteSessionRegistry(tmp_path / "sessions.sqlite3")
    return lambda: FileSessionRegistry(tmp_path / "sessions.txt")


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
