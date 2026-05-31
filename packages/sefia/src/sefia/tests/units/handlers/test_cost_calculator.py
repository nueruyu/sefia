from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from sefia.handlers.cost import CostCalculator, _CostState
from sefia.llm.events import AfterLLMCall
from sefia.llm.messages import LLMResponse


class TestCostCalculator:
    @pytest.fixture
    def mock_context(self, mocker: MockerFixture):
        mock_state_store = mocker.Mock()
        mock_state_store.ensure = AsyncMock(return_value=_CostState(cost=0.0))
        mock_state_store.save = AsyncMock()

        mock_ctx = mocker.Mock()
        mock_ctx.get_state_store.return_value = mock_state_store
        mocker.patch("sefia.handlers.cost.get_context", return_value=mock_ctx)
        return mock_ctx, mock_state_store

    async def test_calculates_and_saves_cost(self, mock_context, mocker: MockerFixture):
        mock_ctx, mock_state_store = mock_context
        mock_cost_per_token = mocker.patch(
            "sefia.handlers.cost.cost_per_token",
            return_value=(0.0005, 0.001),  # 0.0015
        )
        handler = CostCalculator()
        event = AfterLLMCall(
            response=LLMResponse(
                model="gpt-4", usage={"completion_tokens": 50, "prompt_tokens": 100}
            )
        )

        await handler.handle(event)

        mock_cost_per_token.assert_called_once_with(
            model="gpt-4", prompt_tokens=100, completion_tokens=50
        )
        mock_state_store.save.assert_called_once_with(_CostState(cost=0.0015))

    async def test_accumulates_cost(self, mock_context, mocker: MockerFixture):
        mock_ctx, mock_state_store = mock_context
        mock_state_store.ensure.return_value = _CostState(cost=0.01)  # Previous cost
        mocker.patch(
            "sefia.handlers.cost.cost_per_token",
            return_value=(0.0002, 0.0003),  # 0.0005
        )
        handler = CostCalculator()
        event = AfterLLMCall(
            response=LLMResponse(
                model="gpt-4", usage={"completion_tokens": 10, "prompt_tokens": 20}
            )
        )

        await handler.handle(event)

        # 0.01 (previous) + 0.0005 (new)
        mock_state_store.save.assert_called_once_with(_CostState(cost=0.0105))

    async def test_ignores_event_without_usage_info(
        self, mock_context, mocker: MockerFixture
    ):
        mock_ctx, mock_state_store = mock_context
        mocker.patch("sefia.handlers.cost.cost_per_token")
        handler = CostCalculator()
        event = AfterLLMCall(response=LLMResponse(model="gpt-4", usage=None))

        await handler.handle(event)

        mock_state_store.save.assert_not_called()
