from dataclasses import dataclass

from sefia._context import get_context
from sefia.event_system import EventHandler
from sefia.llm.events import AfterLLMCall


@dataclass(frozen=True)
class _CostState:
    """Represents the immutable, persisted state of the cumulative cost."""

    cost: float = 0.0


class CostCalculator(EventHandler[AfterLLMCall]):
    """
    Calculates the cumulative cost of LLM calls and persists it via a StateStore.
    This handler is completely stateless. The total cost can be retrieved
    from the session's state store after execution.
    """

    async def handle(self, event: AfterLLMCall):
        ctx = get_context()

        state_store = ctx.get_state_store("sefia_total_cost", _CostState)
        state = await state_store.ensure()
        if event.response.cost is not None and event.response.cost > 0.0:
            new_state = _CostState(cost=state.cost + event.response.cost)
            await state_store.save(new_state)
