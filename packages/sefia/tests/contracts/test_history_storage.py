"""Apply the public history-storage contract to core implementations."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from sefia import HistorySnapshot, HistoryStorage
from sefia.history_storages import GlyffHistoryStorage
from sefia.testing import HistoryStorageContract, MemoryHistoryStorage


class TestMemoryHistoryStorageContract(HistoryStorageContract):
    @pytest.fixture
    def history_storage(self) -> HistoryStorage:
        return MemoryHistoryStorage()


class TestGlyffHistoryStorageContract(HistoryStorageContract):
    @pytest.fixture
    def history_storage(self, mocker: MockerFixture) -> HistoryStorage:
        stored: HistorySnapshot | None = None
        context = MagicMock()

        async def get(*_args: Any, **_kwargs: Any) -> HistorySnapshot | None:
            return stored

        async def set_value(
            _key: str, value: HistorySnapshot, _type_hint: type[HistorySnapshot]
        ) -> None:
            nonlocal stored
            stored = value

        @asynccontextmanager
        async def transaction() -> AsyncGenerator[None]:
            yield

        context.metadata.get.side_effect = get
        context.metadata.set.side_effect = set_value
        context.get_transaction_scope.side_effect = transaction
        mocker.patch(
            "sefia.history_storages._glyff.glyff.get_context", return_value=context
        )
        return GlyffHistoryStorage()
