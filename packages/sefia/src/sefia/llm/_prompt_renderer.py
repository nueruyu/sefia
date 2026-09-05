from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..inference import FunctionInfo, HistoryItem, ToolCallResult
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
    """Renders decision prompts and tool-result message content as text."""

    @abstractmethod
    def render(self, prompt: DecisionPrompt) -> str: ...

    @abstractmethod
    def render_tool_result(self, result: ToolCallResult) -> str: ...


__all__ = [
    "DecisionPrompt",
    "PromptRenderer",
    "RejectedDecision",
]
