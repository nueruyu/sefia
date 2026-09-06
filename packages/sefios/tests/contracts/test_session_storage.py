"""Apply the public session-storage contract to every built-in implementation."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import sefios.storage as implementations
from glyff import Serializer
from sefios.storage import (
    FileSessionStorage,
    MemorySessionStorage,
    SessionStorage,
    SQLiteSessionStorage,
)
from sefios.testing import SessionStorageContract, SessionStorageFactory
from typing_extensions import override


def _memory(path: Path, serializer: Serializer) -> SessionStorageFactory:
    store = MemorySessionStorage(serializer)
    return lambda: store


def _file(path: Path, serializer: Serializer) -> SessionStorageFactory:
    return lambda: FileSessionStorage(path, serializer)


def _sqlite(path: Path, serializer: Serializer) -> SessionStorageFactory:
    return lambda: SQLiteSessionStorage(path / "state.sqlite3", "session", serializer)


CASE_FACTORIES: dict[
    type[SessionStorage], Callable[[Path, Serializer], SessionStorageFactory]
] = {
    MemorySessionStorage: _memory,
    FileSessionStorage: _file,
    SQLiteSessionStorage: _sqlite,
}


class TestSessionStorageContract(SessionStorageContract):
    _factory: SessionStorageFactory

    @pytest.fixture(
        autouse=True,
        params=tuple(CASE_FACTORIES),
        ids=[cls.__name__ for cls in CASE_FACTORIES],
    )
    def _prepare_store(
        self, request: pytest.FixtureRequest, tmp_path: Path, serializer: Serializer
    ) -> None:
        implementation = cast(type[SessionStorage], request.param)
        self._factory = CASE_FACTORIES[implementation](tmp_path, serializer)

    @override
    def make_session_storage(self) -> SessionStorage:
        return self._factory()


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in implementations.__all__
        if isinstance(value := getattr(implementations, name), type)
        and value is not SessionStorage
        and issubclass(value, SessionStorage)
    }

    assert set(CASE_FACTORIES) == exported
