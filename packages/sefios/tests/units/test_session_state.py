from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from glyff import DomainId, ExecutionId
from glyff.testing import make_execution_id
from pytest_mock import MockerFixture

from sefios._session_state import _SessionState
from sefios.storage import MemorySessionStorage


@dataclass
class StateA:
    value: str


@dataclass
class StateB:
    count: int


def _execution_id(
    name: str, digest: str, parent_id: ExecutionId | None = None
) -> ExecutionId:
    return make_execution_id(
        name,
        parent=parent_id,
        domain_id=DomainId("sefios.tests"),
        arguments={"digest": digest},
    )


@pytest.fixture
def session_state(memory_session_storage: MemorySessionStorage) -> _SessionState:
    return _SessionState(storage=memory_session_storage)


@pytest.fixture
def glyff_ctx(mocker: MockerFixture) -> MagicMock:
    context = MagicMock()
    mocker.patch("sefios._session_state.get_glyff_context", return_value=context)
    return context


class TestSessionState:
    async def test_get_call_state_store_reopens_state_for_same_call_and_suffix(
        self,
        session_state: _SessionState,
        memory_session_storage: MemorySessionStorage,
        glyff_ctx: MagicMock,
    ) -> None:
        glyff_ctx.current_execution_id = _execution_id("Input.get_input", "prompt-a")
        state = StateA("stored")

        await session_state.get_call_state_store("my_state", StateA).save(state)

        reopened = _SessionState(memory_session_storage)
        assert await reopened.get_call_state_store("my_state", StateA).get() == state
        assert await reopened.get_call_state_store("other_state", StateA).get() is None

    async def test_get_call_state_store_separates_argument_scopes(
        self,
        session_state: _SessionState,
        memory_session_storage: MemorySessionStorage,
        glyff_ctx: MagicMock,
    ) -> None:
        first_execution_id = _execution_id("Input.get_input", "prompt-a")
        second_execution_id = _execution_id("Input.get_input", "prompt-b")
        first_state = StateA("first")

        glyff_ctx.current_execution_id = first_execution_id
        await session_state.get_call_state_store("my_state", StateA).save(first_state)

        reopened = _SessionState(memory_session_storage)
        glyff_ctx.current_execution_id = second_execution_id
        assert await reopened.get_call_state_store("my_state", StateA).get() is None
        glyff_ctx.current_execution_id = first_execution_id
        assert (
            await reopened.get_call_state_store("my_state", StateA).get() == first_state
        )

    async def test_get_call_state_store_separates_parent_scopes(
        self,
        session_state: _SessionState,
        memory_session_storage: MemorySessionStorage,
        glyff_ctx: MagicMock,
    ) -> None:
        first_parent = _execution_id(
            "RequirementsClarifier.clarify_request", "clarifier"
        )
        second_parent = _execution_id("NewsWriter.write_article", "writer")
        first_execution_id = _execution_id(
            "Input.get_input", "same-prompt", first_parent
        )
        second_execution_id = _execution_id(
            "Input.get_input", "same-prompt", second_parent
        )
        first_state = StateA("first")

        glyff_ctx.current_execution_id = first_execution_id
        await session_state.get_call_state_store("my_state", StateA).save(first_state)

        reopened = _SessionState(memory_session_storage)
        glyff_ctx.current_execution_id = second_execution_id
        assert await reopened.get_call_state_store("my_state", StateA).get() is None
        glyff_ctx.current_execution_id = first_execution_id
        assert (
            await reopened.get_call_state_store("my_state", StateA).get() == first_state
        )

    def test_get_call_state_store_raises_error_outside_engrave(
        self, session_state: _SessionState, glyff_ctx: MagicMock
    ) -> None:
        glyff_ctx.current_execution_id = None

        with pytest.raises(RuntimeError, match="can only be used inside"):
            session_state.get_call_state_store("my_state", StateA)

    def test_get_state_store_returns_same_instance_for_same_key(
        self, session_state: _SessionState
    ) -> None:
        store1 = session_state.get_state_store("shared_key", StateA)
        store2 = session_state.get_state_store("shared_key", StateA)

        assert store1 is store2

    def test_get_state_store_raises_type_error_on_mismatch(
        self, session_state: _SessionState
    ) -> None:
        session_state.get_state_store("shared_key", StateA)

        with pytest.raises(
            TypeError, match="was already created with a different type"
        ):
            session_state.get_state_store("shared_key", StateB)
