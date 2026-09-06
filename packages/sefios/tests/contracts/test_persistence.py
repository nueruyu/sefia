"""Apply the public persistence contract to every built-in provider."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import sefios
from sefios import (
    FilePersistence,
    MemoryPersistence,
    PersistenceProvider,
    SQLitePersistence,
)
from sefios.testing import PersistenceProviderContract
from typing_extensions import override

CASE_FACTORIES: dict[
    type[PersistenceProvider], Callable[[Path], PersistenceProvider]
] = {
    MemoryPersistence: lambda path: MemoryPersistence(),
    FilePersistence: lambda path: FilePersistence(path / "sessions"),
    SQLitePersistence: lambda path: SQLitePersistence(path / "sessions.sqlite3"),
}


class TestPersistenceProviderContract(PersistenceProviderContract):
    _provider: PersistenceProvider

    @pytest.fixture(
        autouse=True,
        params=tuple(CASE_FACTORIES),
        ids=[cls.__name__ for cls in CASE_FACTORIES],
    )
    def _prepare_provider(self, request: pytest.FixtureRequest, tmp_path: Path) -> None:
        implementation = cast(type[PersistenceProvider], request.param)
        self._provider = CASE_FACTORIES[implementation](tmp_path)

    @override
    def make_persistence_provider(self) -> PersistenceProvider:
        return self._provider


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in sefios.__all__
        if isinstance(value := getattr(sefios, name), type)
        and value is not PersistenceProvider
        and issubclass(value, PersistenceProvider)
    }

    assert set(CASE_FACTORIES) == exported
