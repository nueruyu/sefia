from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from glyff import ExecutionId

from sefios._session_state import _SessionState


@dataclass
class StateA:
    value: str


@dataclass
class StateB:
    count: int


@pytest.fixture
def session_state():
    return _SessionState(storage=MagicMock())


class TestSessionState:
    def test_get_call_state_store_creates_scoped_key(self, session_state, mocker):
        # Arrange
        execution_id = ExecutionId(
            parent_id=None,
            name="HumanInputTool.get_human_input",
            sequence=0,
            args_hash="question-a",
        )
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
        first_execution_id = ExecutionId(
            parent_id=None,
            name="HumanInputTool.get_human_input",
            sequence=0,
            args_hash="question-a",
        )
        second_execution_id = ExecutionId(
            parent_id=None,
            name="HumanInputTool.get_human_input",
            sequence=0,
            args_hash="question-b",
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

    def test_get_call_state_store_includes_parent_scope(self, session_state, mocker):
        # Arrange
        first_parent = ExecutionId(
            parent_id=None,
            name="RequirementsClarifier.clarify_request",
            sequence=0,
            args_hash="clarifier",
        )
        second_parent = ExecutionId(
            parent_id=None,
            name="NewsWriter.write_article",
            sequence=0,
            args_hash="writer",
        )
        first_execution_id = ExecutionId(
            parent_id=first_parent,
            name="HumanInputTool.get_human_input",
            sequence=0,
            args_hash="same-question",
        )
        second_execution_id = ExecutionId(
            parent_id=second_parent,
            name="HumanInputTool.get_human_input",
            sequence=0,
            args_hash="same-question",
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
