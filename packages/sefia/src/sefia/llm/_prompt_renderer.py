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
    """Renders text representations that a transport sends to an LLM."""

    @abstractmethod
    def render(self, prompt: DecisionPrompt) -> str: ...

    def render_tool_result(self, result: object) -> str:
        """Render a tool result as prompt text for native decision history.

        Renderers used only by structured or prompted transports need not override
        this method.
        """
        raise NotImplementedError(
            "PromptRenderer.render_tool_result() is required by "
            "NativeDecisionTransport."
        )


__all__ = [
    "DecisionPrompt",
    "PromptRenderer",
    "RejectedDecision",
]
