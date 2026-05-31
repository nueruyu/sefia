from typing import Type

from pydantic import BaseModel

from sefia.context import get_context
from sefia.events import Event
from sefia.interfaces import EventHandler
from sefia.llm.events import AfterLLMCall


class _CostState(BaseModel, frozen=True):
    """Represents the immutable, persisted state of the cumulative cost."""

    cost: float = 0.0


class CostCalculator(EventHandler[AfterLLMCall]):
    """
    Calculates the cumulative cost of LLM calls and persists it via a StateStore.
    This handler is completely stateless. The total cost can be retrieved
    from the session's state store after execution.
    """

    @property
    def event_types(self) -> tuple[Type[Event], ...]:
        return (AfterLLMCall,)

    async def handle(self, event: AfterLLMCall):
        ctx = get_context()

        state_store = ctx.get_state_store("sefia_total_cost", _CostState)
        state = await state_store.ensure()
        if event.response.cost is not None and event.response.cost > 0.0:
            new_state = _CostState(cost=state.cost + event.response.cost)
            await state_store.save(new_state)
