from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..exceptions import InvalidInferenceResponseError
from ..inference import FunctionInfo, HistoryItem
from ._messages import Message
from .step_decision import StepDecisionSpec


class PromptRenderer(ABC):
    """Renders the messages used for an inference call."""

    @abstractmethod
    def render(
        self,
        function_info: FunctionInfo,
        history: Sequence[HistoryItem],
        decision_spec: StepDecisionSpec,
    ) -> list[Message]:
        """Render the initial messages for an inference call."""
        ...

    @abstractmethod
    def render_repair(self, error: InvalidInferenceResponseError) -> list[Message]:
        """Render corrective messages for an invalid response."""
        ...
