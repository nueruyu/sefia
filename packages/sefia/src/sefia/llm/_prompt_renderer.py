from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..inference import FunctionInfo, HistoryItem
from .step_decision import StepTool


@dataclass(frozen=True)
class RejectedDecision:
    content: str | None
    reason: str


@dataclass(frozen=True)
class DecisionPrompt:
    function: FunctionInfo
    tools: tuple[StepTool, ...]
    history: tuple[HistoryItem, ...]
    response_instructions: str
    rejected: RejectedDecision | None = None


class PromptRenderer(ABC):
    """Renders a complete inference prompt as text."""

    @abstractmethod
    def render(self, prompt: DecisionPrompt) -> str: ...

    @abstractmethod
    def render_tool_result(self, result: object) -> str:
        """Render a tool result as prompt text."""
        ...


__all__ = [
    "DecisionPrompt",
    "PromptRenderer",
    "RejectedDecision",
]
