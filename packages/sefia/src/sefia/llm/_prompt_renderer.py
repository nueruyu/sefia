from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..inference import FunctionInfo
from .step_decision import StepTool


@dataclass(frozen=True)
class RejectedDecision:
    content: str | None
    reason: str


@dataclass(frozen=True)
class InferencePrompt:
    function: FunctionInfo
    arguments: Mapping[str, Any]
    tools: tuple[StepTool, ...]


class PromptRenderer(ABC):
    """Renders the standard inference prompt from function data and tools."""

    @abstractmethod
    def render(self, prompt: InferencePrompt) -> str: ...


__all__ = [
    "InferencePrompt",
    "PromptRenderer",
    "RejectedDecision",
]
