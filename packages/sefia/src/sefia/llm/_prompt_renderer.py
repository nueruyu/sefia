from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..inference import FunctionInfo
from .step_decision import StepTool
from .structured_data import StructuredData


@dataclass(frozen=True)
class InferencePrompt:
    function: FunctionInfo
    arguments: StructuredData
    tools: tuple[StepTool, ...]


class PromptRenderer(ABC):
    """Renders the standard inference prompt from function data and tools."""

    @abstractmethod
    def render(self, prompt: InferencePrompt) -> str: ...


__all__ = [
    "InferencePrompt",
    "PromptRenderer",
]
