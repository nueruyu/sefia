from dataclasses import dataclass
from typing import Type

from sefia.event_system import Event, EventHandler
from sefia.llm.events import AfterLLMCall

from ..state import get_state, state


@state(key="sefios.cost")
@dataclass(frozen=True)
class CostState:
    """Represents the immutable, persisted state of the cumulative cost."""

    cost: float = 0.0


class CostCalculator(EventHandler[AfterLLMCall]):
    """
    Calculates the cumulative cost of LLM calls and persists it via the state
    container. This handler is completely stateless. The total cost can be
    retrieved from the session's state container after execution.
    """

    @property
    def event_types(self) -> tuple[Type[Event], ...]:
        return (AfterLLMCall,)

    async def handle(self, event: AfterLLMCall):
        state_store = get_state().get(CostState)
        current = await state_store.ensure()
        if event.response.cost is not None and event.response.cost > 0.0:
            new_state = CostState(cost=current.cost + event.response.cost)
            await state_store.save(new_state)
