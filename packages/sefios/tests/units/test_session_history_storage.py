from unittest.mock import MagicMock

import pytest
from glyff import ExecutionId
from sefia import HistorySnapshot
from sefia.inference import ToolCallDecision, ToolCallRequest, ToolCallResult

from sefios import MemorySessionStorage, SessionHistoryStorage
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


class TestSessionHistoryStorage:
    async def test_round_trips_a_snapshot(self, storage, glyff_ctx):
        history_storage = SessionHistoryStorage()
        snapshot = HistorySnapshot(
            items=(
                ToolCallDecision(
                    calls=[ToolCallRequest(id="1", name="add_note", arguments={"x": 1})]
                ),
                ToolCallResult(tool_call_id="1", result="noted"),
            ),
            completed_steps=1,
        )

        with bind_session_storage(storage):
            await history_storage.save(snapshot)
            loaded = await history_storage.load()

        assert loaded == snapshot

    async def test_load_returns_empty_snapshot_when_nothing_stored(
        self, storage, glyff_ctx
    ):
        with bind_session_storage(storage):
            assert await SessionHistoryStorage().load() == HistorySnapshot()

    async def test_histories_are_scoped_per_run_execution(self, storage, glyff_ctx):
        history_storage = SessionHistoryStorage()
        first = HistorySnapshot(
            items=(ToolCallResult(tool_call_id="1", result="first"),),
            completed_steps=1,
        )

        with bind_session_storage(storage):
            glyff_ctx.current_execution_id = _execution_id("run-a")
            await history_storage.save(first)
            glyff_ctx.current_execution_id = _execution_id("run-b")
            assert await history_storage.load() == HistorySnapshot()
            glyff_ctx.current_execution_id = _execution_id("run-a")
            assert await history_storage.load() == first

    async def test_raises_outside_an_engraved_run(self, storage, glyff_ctx):
        glyff_ctx.current_execution_id = None
        with bind_session_storage(storage):
            with pytest.raises(RuntimeError, match="engraved"):
                await SessionHistoryStorage().load()
