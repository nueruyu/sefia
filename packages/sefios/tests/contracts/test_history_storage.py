"""Apply the core history-storage contract to the sefios implementation."""

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from glyff import ArgumentsDigest, DomainId, ExecutionId, ExecutionName, Serializer
from pytest_mock import MockerFixture

from sefia import HistoryStorage
from sefia.testing import HistoryStorageContract

from sefios import MemorySessionStorage
from sefios._session_state import bind_session_storage
from sefios.history_storages import SessionHistoryStorage


def _execution_id() -> ExecutionId:
    return ExecutionId(
        parent_id=None,
        domain_id=DomainId("sefios.tests"),
        name=ExecutionName("Agent.chat"),
        sequence=0,
        arguments_digest=ArgumentsDigest("history-contract"),
    )


class TestSessionHistoryStorageContract(HistoryStorageContract):
    @pytest.fixture
    def history_storage(
        self,
        mocker: MockerFixture,
        serializer: Serializer,
    ) -> Iterator[HistoryStorage]:
        context = MagicMock()
        context.current_execution_id = _execution_id()
        mocker.patch(
            "sefios.history_storages._session.get_glyff_context", return_value=context
        )
        with bind_session_storage(MemorySessionStorage(serializer)):
            yield SessionHistoryStorage()
