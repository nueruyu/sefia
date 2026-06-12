from dataclasses import dataclass
from typing import Type

from sefia import EventHandler
from sefia._context import get_context
from sefia.events import Event
from sefia.llm.events import AfterLLMCall


@dataclass(frozen=True)
class _CostState:
    cost: float = 0.0


class CostCalculator(EventHandler[AfterLLMCall]):
    @property
    def event_types(self) -> tuple[Type[Event], ...]:
        return (AfterLLMCall,)

    async def handle(self, event: AfterLLMCall):
        ctx = get_context()
        state_store = ctx.get_state_store("sefia_total_cost", _CostState)
        state = await state_store.ensure()
        if event.response.cost is not None and event.response.cost > 0.0:
            await state_store.save(_CostState(cost=state.cost + event.response.cost))
