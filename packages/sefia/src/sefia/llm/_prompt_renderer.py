from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..inference import FunctionInfo, HistoryItem
from .json_schema import JsonValue
from .step_decision import DecisionSpec


@dataclass(frozen=True)
class RejectedDecision:
    content: str | None
    reason: str


@dataclass(frozen=True)
class DecisionResponseForm:
    label: str
    example: str
    schema: JsonValue | None = None


@dataclass(frozen=True)
class DecisionResponseInstructions:
    forms: tuple[DecisionResponseForm, ...]
    rules: tuple[str, ...]


@dataclass(frozen=True)
class DecisionPrompt:
    function: FunctionInfo
    decision: DecisionSpec
    history: tuple[HistoryItem, ...]
    response: DecisionResponseInstructions
    rejected: RejectedDecision | None = None


class PromptRenderer(ABC):
    """Renders a complete inference prompt as text."""

    @abstractmethod
    def render(self, prompt: DecisionPrompt) -> str: ...


__all__ = [
    "DecisionPrompt",
    "DecisionResponseForm",
    "DecisionResponseInstructions",
    "PromptRenderer",
    "RejectedDecision",
]
