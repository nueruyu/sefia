"""Apply the public session-registry contract to every built-in implementation."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import sefios.sessions as implementations
from sefios.sessions import (
    FileSessionRegistry,
    MemorySessionRegistry,
    SessionRegistry,
    SQLiteSessionRegistry,
)
from sefios.testing import SessionRegistryContract, SessionRegistryFactory
from typing_extensions import override


def _memory(path: Path) -> SessionRegistryFactory:
    store = MemorySessionRegistry()
    return lambda: store


def _file(path: Path) -> SessionRegistryFactory:
    return lambda: FileSessionRegistry(path / "sessions.txt")


def _sqlite(path: Path) -> SessionRegistryFactory:
    return lambda: SQLiteSessionRegistry(path / "sessions.sqlite3")


CASE_FACTORIES: dict[
    type[SessionRegistry], Callable[[Path], SessionRegistryFactory]
] = {
    MemorySessionRegistry: _memory,
    FileSessionRegistry: _file,
    SQLiteSessionRegistry: _sqlite,
}


class TestSessionRegistryContract(SessionRegistryContract):
    _factory: SessionRegistryFactory

    @pytest.fixture(
        autouse=True,
        params=tuple(CASE_FACTORIES),
        ids=[cls.__name__ for cls in CASE_FACTORIES],
    )
    def _prepare_store(self, request: pytest.FixtureRequest, tmp_path: Path) -> None:
        implementation = cast(type[SessionRegistry], request.param)
        self._factory = CASE_FACTORIES[implementation](tmp_path)

    @override
    def make_session_registry(self) -> SessionRegistry:
        return self._factory()


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in implementations.__all__
        if isinstance(value := getattr(implementations, name), type)
        and value is not SessionRegistry
        and issubclass(value, SessionRegistry)
    }

    assert set(CASE_FACTORIES) == exported
