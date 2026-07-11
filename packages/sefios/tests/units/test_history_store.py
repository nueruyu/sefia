from unittest.mock import MagicMock

import pytest
from glyff import ExecutionId
from sefia.inference import ToolCallDecision, ToolCallRequest, ToolCallResult

from sefios import DurableHistoryStore, MemorySessionStorage
from sefios._session_state import bind_session_storage


def _execution_id(args_hash: str = "hash-a") -> ExecutionId:
    return ExecutionId(
        parent_id=None, name="Agent.chat", sequence=0, args_hash=args_hash
    )


@pytest.fixture
def storage(serializer) -> MemorySessionStorage:
    return MemorySessionStorage(serializer=serializer)


@pytest.fixture
def glyff_ctx(mocker) -> MagicMock:
    ctx = MagicMock()
    ctx.current_execution_id = _execution_id()
    mocker.patch("sefios._history.get_glyff_context", return_value=ctx)
    return ctx


class TestDurableHistoryStore:
    async def test_round_trips_history_items(self, storage, glyff_ctx):
        store = DurableHistoryStore()
        history = [
            ToolCallDecision(
                calls=[ToolCallRequest(id="1", name="add_note", arguments={"x": 1})]
            ),
            ToolCallResult(tool_call_id="1", result="noted"),
        ]

        with bind_session_storage(storage):
            await store.save(history)
            loaded = await store.load()

        assert loaded == history

    async def test_load_returns_empty_when_nothing_stored(self, storage, glyff_ctx):
        with bind_session_storage(storage):
            assert await DurableHistoryStore().load() == []

    async def test_histories_are_scoped_per_run_execution(self, storage, glyff_ctx):
        store = DurableHistoryStore()
        first_run = [ToolCallResult(tool_call_id="1", result="first")]

        with bind_session_storage(storage):
            glyff_ctx.current_execution_id = _execution_id("run-a")
            await store.save(first_run)
            glyff_ctx.current_execution_id = _execution_id("run-b")
            assert await store.load() == []
            glyff_ctx.current_execution_id = _execution_id("run-a")
            assert await store.load() == first_run

    async def test_raises_outside_an_engraved_run(self, storage, glyff_ctx):
        glyff_ctx.current_execution_id = None
        with bind_session_storage(storage):
            with pytest.raises(RuntimeError, match="engraved"):
                await DurableHistoryStore().load()
