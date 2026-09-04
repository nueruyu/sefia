from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..inference import FunctionInfo, HistoryItem
from .step_decision import DecisionSpec


@dataclass(frozen=True)
class RejectedDecision:
    content: str | None
    reason: str


@dataclass(frozen=True)
class DecisionPrompt:
    function: FunctionInfo
    spec: DecisionSpec
    history: tuple[HistoryItem, ...]
    response_instructions: str
    rejected: RejectedDecision | None = None


class PromptRenderer(ABC):
    """Renders a complete inference prompt as text."""

    @abstractmethod
    def render(self, prompt: DecisionPrompt) -> str: ...


__all__ = [
    "DecisionPrompt",
    "PromptRenderer",
    "RejectedDecision",
]
