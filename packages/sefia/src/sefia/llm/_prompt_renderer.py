from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..inference import FunctionInfo
from .json import JsonSnapshot
from .step_decision import StepTool


@dataclass(frozen=True)
class InferencePrompt:
    function: FunctionInfo
    arguments: JsonSnapshot
    tools: tuple[StepTool, ...]


class PromptRenderer(ABC):
    """Renders the standard inference prompt from function data and tools."""

    @abstractmethod
    def render(self, prompt: InferencePrompt) -> str: ...


__all__ = [
    "InferencePrompt",
    "PromptRenderer",
]
