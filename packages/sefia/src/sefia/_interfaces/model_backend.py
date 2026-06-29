from abc import ABC, abstractmethod
from typing import Any, Callable

from .decision_model import DecisionModel, DecisionModelSpec


class ModelBackend(ABC):
    """
    Interface for tool schema generation and LLM decision model construction.
    """

    @abstractmethod
    def get_function_name(self, func: Callable[..., Any]) -> str:
        """Generate the stable tool-call name for a function."""
        ...

    @abstractmethod
    def get_function_schema(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
    ) -> dict:
        """Generate a tool-call schema for a function signature."""
        ...

    @abstractmethod
    def build_decision_model(self, spec: DecisionModelSpec) -> DecisionModel:
        """Build a decision model for a structured LLM response."""
        ...
