from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from glyff import ArgumentsDigest, DomainId, ExecutionId, ExecutionName, Serializer
from pytest_mock import MockerFixture
from sefia import HistorySnapshot, HistoryStorage
from sefia.history_storages import GlyffHistoryStorage
from sefia.inference import ToolCallResult
from sefia.testing import MemoryHistoryStorage

from sefios import MemorySessionStorage
from sefios._session_state import bind_session_storage
from sefios.history_storages import SessionHistoryStorage

HISTORY_STORAGE_TYPES = (
    GlyffHistoryStorage,
    MemoryHistoryStorage,
    SessionHistoryStorage,
)


def _execution_id() -> ExecutionId:
    return ExecutionId(
        parent_id=None,
        domain_id=DomainId("sefios.tests"),
        name=ExecutionName("Agent.chat"),
        sequence=0,
        arguments_digest=ArgumentsDigest("history-contract"),
    )


@pytest.fixture(
    params=HISTORY_STORAGE_TYPES, ids=lambda storage_type: storage_type.__name__
)
def history_storage(
    request: pytest.FixtureRequest,
    mocker: MockerFixture,
    serializer: Serializer,
) -> Iterator[HistoryStorage]:
    storage_type = request.param
    if storage_type is MemoryHistoryStorage:
        yield storage_type()
        return

    if storage_type is SessionHistoryStorage:
        ctx = MagicMock()
        ctx.current_execution_id = _execution_id()
        mocker.patch(
            "sefios.history_storages._session.get_glyff_context", return_value=ctx
        )
        with bind_session_storage(MemorySessionStorage(serializer)):
            yield storage_type()
        return

    stored: HistorySnapshot | None = None
    ctx = MagicMock()

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

    ctx.metadata.get.side_effect = get
    ctx.metadata.set.side_effect = set_value
    ctx.get_transaction_scope.side_effect = transaction
    mocker.patch("sefia.history_storages._glyff.glyff.get_context", return_value=ctx)
    yield storage_type()


def _snapshot(value: str, completed_steps: int) -> HistorySnapshot:
    return HistorySnapshot(
        items=(ToolCallResult(tool_call_id="call-1", result=value),),
        completed_steps=completed_steps,
    )


async def test_contract_loads_an_empty_snapshot_initially(
    history_storage: HistoryStorage,
) -> None:
    assert await history_storage.load() == HistorySnapshot()


async def test_contract_round_trips_and_replaces_the_latest_snapshot(
    history_storage: HistoryStorage,
) -> None:
    first = _snapshot("first", 1)
    second = _snapshot("second", 2)

    await history_storage.save(first)
    assert await history_storage.load() == first

    await history_storage.save(second)
    assert await history_storage.load() == second
