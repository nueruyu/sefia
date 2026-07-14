from abc import ABC, abstractmethod
from collections.abc import Sequence

from .._tool_system import ToolRegistry
from ..event_system import EventPublisher
from ..inference import FunctionInfo, HistoryItem, InferenceDecision


class InferenceStrategy(ABC):
    """
    Protocol for a strategy that decides the next step in an inference process.
    """

    @abstractmethod
    async def decide_next_step(
        self,
        function_info: FunctionInfo,
        history: Sequence[HistoryItem],
        tools: ToolRegistry,
        publisher: EventPublisher,
    ) -> InferenceDecision:
        """Decides the next action, either calling tools or returning a result."""
        ...
