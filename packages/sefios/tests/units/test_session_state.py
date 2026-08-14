from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from glyff import ArgumentsDigest, DomainId, ExecutionId, ExecutionName

from sefios._session_state import _SessionState


@dataclass
class StateA:
    value: str


@dataclass
class StateB:
    count: int


def _execution_id(
    name: str, digest: str, parent_id: ExecutionId | None = None
) -> ExecutionId:
    return ExecutionId(
        parent_id=parent_id,
        domain_id=DomainId("sefios.tests"),
        name=ExecutionName(name),
        sequence=0,
        arguments_digest=ArgumentsDigest(digest),
    )


@pytest.fixture
def session_state():
    return _SessionState(storage=MagicMock())


class TestSessionState:
    def test_get_call_state_store_creates_scoped_key(self, session_state, mocker):
        # Arrange
        execution_id = _execution_id("Input.get_input", "prompt-a")
        mock_glyff_ctx = MagicMock()
        mock_glyff_ctx.current_execution_id = execution_id
        mocker.patch(
            "sefios._session_state.get_glyff_context", return_value=mock_glyff_ctx
        )

        # Act
        store = session_state.get_call_state_store("my_state", StateA)

        # Assert
        assert store._key.startswith("call_state/")
        assert store._key.endswith("/my_state")
        assert str(execution_id) not in store._key

    def test_get_call_state_store_includes_args_hash_in_scope(
        self, session_state, mocker
    ):
        # Arrange
        first_execution_id = _execution_id("Input.get_input", "prompt-a")
        second_execution_id = _execution_id("Input.get_input", "prompt-b")
        mock_glyff_ctx = MagicMock()
        mocker.patch(
            "sefios._session_state.get_glyff_context", return_value=mock_glyff_ctx
        )

        # Act
        mock_glyff_ctx.current_execution_id = first_execution_id
        first_store = session_state.get_call_state_store("my_state", StateA)
        mock_glyff_ctx.current_execution_id = second_execution_id
        second_store = session_state.get_call_state_store("my_state", StateA)

        # Assert
        assert first_store._key != second_store._key

    def test_get_call_state_store_includes_parent_scope(self, session_state, mocker):
        # Arrange
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
        mock_glyff_ctx = MagicMock()
        mocker.patch(
            "sefios._session_state.get_glyff_context", return_value=mock_glyff_ctx
        )

        # Act
        mock_glyff_ctx.current_execution_id = first_execution_id
        first_store = session_state.get_call_state_store("my_state", StateA)
        mock_glyff_ctx.current_execution_id = second_execution_id
        second_store = session_state.get_call_state_store("my_state", StateA)

        # Assert
        assert first_store._key != second_store._key

    def test_get_call_state_store_raises_error_outside_engrave(
        self, session_state, mocker
    ):
        # Arrange: simulate being outside an engraved call (no current execution).
        mock_glyff_ctx = MagicMock()
        mock_glyff_ctx.current_execution_id = None
        mocker.patch(
            "sefios._session_state.get_glyff_context", return_value=mock_glyff_ctx
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="can only be used inside"):
            session_state.get_call_state_store("my_state", StateA)

    def test_get_state_store_returns_same_instance_for_same_key(self, session_state):
        store1 = session_state.get_state_store("shared_key", StateA)
        store2 = session_state.get_state_store("shared_key", StateA)

        assert store1 is store2

    def test_get_state_store_raises_type_error_on_mismatch(self, session_state):
        session_state.get_state_store("shared_key", StateA)

        with pytest.raises(
            TypeError, match="was already created with a different type"
        ):
            session_state.get_state_store("shared_key", StateB)
