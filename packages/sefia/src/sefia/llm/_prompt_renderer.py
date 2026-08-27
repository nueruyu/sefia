from abc import ABC, abstractmethod

from ..exceptions import InvalidInferenceResponseError
from ..inference import FunctionInfo
from .step_decision import StepDecisionSpec


class PromptRenderer(ABC):
    """Renders the text content used in inference prompts."""

    @abstractmethod
    def render_instructions(
        self,
        function_info: FunctionInfo,
        decision_spec: StepDecisionSpec,
    ) -> str:
        """Render the instructions governing an inference call."""
        ...

    @abstractmethod
    def render_invocation(self, function_info: FunctionInfo) -> str:
        """Render the invocation context for an inference call."""
        ...

    @abstractmethod
    def render_response_feedback(self, error: InvalidInferenceResponseError) -> str:
        """Render corrective feedback for an invalid response."""
        ...
