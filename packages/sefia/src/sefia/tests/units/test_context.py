import uuid
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from sefia.context import InferenceContext


class StateA(BaseModel):
    value: str


class StateB(BaseModel):
    count: int


@pytest.fixture
def mock_glyff_session():
    return MagicMock()


@pytest.fixture
def inference_context(mock_glyff_session):
    return InferenceContext(
        glyff_session=mock_glyff_session,
        session_store=MagicMock(),
        llm_client=MagicMock(),
        inference_strategy=MagicMock(),
        policies=[],
        tool_collector=MagicMock(),
    )


class TestInferenceContext:
    def test_get_call_state_store_creates_scoped_key(self, inference_context, mocker):
        # Arrange
        execution_id = uuid.uuid4()
        mock_glyff_ctx = MagicMock()
        mock_glyff_ctx.current_execution_id = execution_id
        mocker.patch("sefia.context.get_glyff_context", return_value=mock_glyff_ctx)

        # Act
        store = inference_context.get_call_state_store("my_state", StateA)

        # Assert
        expected_key = f"call_state::{str(execution_id)}::my_state"
        assert store._key == expected_key

    def test_get_call_state_store_raises_error_outside_engrave(
        self, inference_context, mocker
    ):
        # Arrange
        # Simulate being outside an engraved function by making current_execution_id None
        mock_glyff_ctx = MagicMock()
        mock_glyff_ctx.current_execution_id = None
        mocker.patch("sefia.context.get_glyff_context", return_value=mock_glyff_ctx)

        # Act & Assert
        with pytest.raises(RuntimeError, match="can only be used inside"):
            inference_context.get_call_state_store("my_state", StateA)

    def test_get_state_store_returns_same_instance_for_same_key(
        self, inference_context
    ):
        # Act
        store1 = inference_context.get_state_store("shared_key", StateA)
        store2 = inference_context.get_state_store("shared_key", StateA)

        # Assert
        assert store1 is store2

    def test_get_state_store_raises_type_error_on_mismatch(self, inference_context):
        # Arrange
        inference_context.get_state_store("shared_key", StateA)

        # Act & Assert
        with pytest.raises(
            TypeError, match="was already created with a different type"
        ):
            inference_context.get_state_store("shared_key", StateB)
