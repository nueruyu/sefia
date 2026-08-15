from unittest.mock import AsyncMock, Mock

import pytest
from pytest_mock import MockerFixture

from sefia.llm import LLMResponse
from sefia.llm.events import AfterLLMCall
from sefios.handlers import CostCalculator, CostState


class TestCostCalculator:
    @pytest.fixture
    def mock_state_store(self, mocker: MockerFixture) -> Mock:
        store = mocker.Mock()
        store.ensure = AsyncMock(return_value=CostState(cost=0.0))
        store.save = AsyncMock()

        container = mocker.Mock()
        container.get.return_value = store
        mocker.patch("sefios.handlers._cost.get_state", return_value=container)
        return store

    async def test_calculates_and_saves_cost(self, mock_state_store: Mock) -> None:
        handler = CostCalculator()
        event = AfterLLMCall(
            LLMResponse(
                model="gpt-4",
                usage={"completion_tokens": 50, "prompt_tokens": 100},
                cost=0.0015,
            )
        )

        await handler.handle(event)

        mock_state_store.save.assert_called_once_with(CostState(cost=0.0015))

    async def test_accumulates_cost(self, mock_state_store: Mock) -> None:
        mock_state_store.ensure.return_value = CostState(cost=0.01)
        handler = CostCalculator()
        event = AfterLLMCall(
            LLMResponse(
                model="gpt-4",
                usage={"completion_tokens": 10, "prompt_tokens": 20},
                cost=0.0005,
            )
        )

        await handler.handle(event)

        mock_state_store.save.assert_called_once_with(CostState(cost=0.0105))

    async def test_ignores_event_without_cost_info(
        self, mock_state_store: Mock
    ) -> None:
        handler = CostCalculator()
        event = AfterLLMCall(LLMResponse(model="gpt-4", cost=None))

        await handler.handle(event)

        mock_state_store.save.assert_not_called()
