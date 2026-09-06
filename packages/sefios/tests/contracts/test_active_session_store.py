"""Apply the public active-session-store contract to every built-in implementation."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import sefios.sessions as implementations
from sefios.sessions import (
    ActiveSessionStore,
    FileActiveSessionStore,
    MemoryActiveSessionStore,
)
from sefios.testing import ActiveSessionStoreContract, ActiveSessionStoreFactory
from typing_extensions import override


def _memory(path: Path) -> ActiveSessionStoreFactory:
    store = MemoryActiveSessionStore()
    return lambda: store


def _file(path: Path) -> ActiveSessionStoreFactory:
    return lambda: FileActiveSessionStore(path / "active-session.txt")


CASE_FACTORIES: dict[
    type[ActiveSessionStore], Callable[[Path], ActiveSessionStoreFactory]
] = {
    MemoryActiveSessionStore: _memory,
    FileActiveSessionStore: _file,
}


class TestActiveSessionStoreContract(ActiveSessionStoreContract):
    _factory: ActiveSessionStoreFactory

    @pytest.fixture(
        autouse=True,
        params=tuple(CASE_FACTORIES),
        ids=[cls.__name__ for cls in CASE_FACTORIES],
    )
    def _prepare_store(self, request: pytest.FixtureRequest, tmp_path: Path) -> None:
        implementation = cast(type[ActiveSessionStore], request.param)
        self._factory = CASE_FACTORIES[implementation](tmp_path)

    @override
    def make_active_session_store(self) -> ActiveSessionStore:
        return self._factory()


def test_contract_covers_all_exported_implementations() -> None:
    exported = {
        value
        for name in implementations.__all__
        if isinstance(value := getattr(implementations, name), type)
        and value is not ActiveSessionStore
        and issubclass(value, ActiveSessionStore)
    }

    assert set(CASE_FACTORIES) == exported
