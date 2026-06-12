from abc import ABC, abstractmethod
from typing import Any

from .._event_publisher import EventPublisher
from ..inference import HistoryItem, InferenceDecision


class InferenceStrategy(ABC):
    """
    Protocol for a strategy that decides the next step in an inference process.
    """

    @abstractmethod
    async def decide_next_step(
        self,
        instructions: str,
        arguments: dict[str, Any],
        argument_type_hints: dict[str, Any],
        history: list[HistoryItem],
        tools: list[dict],
        output_type: Any,
        publisher: EventPublisher,
    ) -> InferenceDecision:
        """Decides the next action, either calling tools or returning a final answer."""
        ...
