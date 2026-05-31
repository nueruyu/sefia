import logging
from typing import Type

from litellm import cost_per_token
from pydantic import BaseModel

from sefia.context import get_context
from sefia.events import Event
from sefia.interfaces import EventHandler
from sefia.llm.events import AfterLLMCall

logger = logging.getLogger(__name__)


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

        if not (event.response.usage and event.response.model):
            return

        try:
            prompt_cost, completion_cost = cost_per_token(
                model=event.response.model,
                prompt_tokens=event.response.usage.get("prompt_tokens", 0),
                completion_tokens=event.response.usage.get("completion_tokens", 0),
            )

            total_cost = prompt_cost + completion_cost
            if total_cost > 0.0:
                new_state = _CostState(cost=state.cost + total_cost)
                await state_store.save(new_state)
        except Exception:
            logger.warning(
                "Failed to calculate cost for model %s",
                event.response.model,
                exc_info=True,
            )
