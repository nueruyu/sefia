from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..inference import FunctionInfo, HistoryItem, ToolCallResult
from .step_decision import StepTool


@dataclass(frozen=True)
class RejectedDecision:
    content: str | None
    reason: str


@dataclass(frozen=True)
class DecisionPrompt:
    function: FunctionInfo
    arguments: Mapping[str, Any]
    tools: tuple[StepTool, ...]


class PromptRenderer(ABC):
    """Renders decision prompts and tool-result message content as text."""

    @abstractmethod
    def render(self, prompt: DecisionPrompt) -> str: ...

    @abstractmethod
    def render_decision_instructions(self, instructions: str) -> str: ...

    @abstractmethod
    def render_history(self, history: tuple[HistoryItem, ...]) -> str: ...

    @abstractmethod
    def render_rejection(self, rejected: RejectedDecision) -> str: ...

    @abstractmethod
    def render_tool_result(self, result: ToolCallResult) -> str: ...


__all__ = [
    "DecisionPrompt",
    "PromptRenderer",
    "RejectedDecision",
]
