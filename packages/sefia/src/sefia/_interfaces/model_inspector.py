from abc import ABC, abstractmethod
from typing import Any, Callable, Type

from .decision_model import DecisionModel, DecisionModelSpec


class ModelInspector(ABC):
    """
    Interface for schema generation and model validation.
    """

    @abstractmethod
    def get_type_schema(self, model_type: Type[Any] | Any) -> dict:
        """Generate a JSON schema for a target type."""
        ...

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
    def validate(self, model_type: Type[Any] | Any, data: Any) -> Any:
        """
        Validate arbitrary input data against a target type and return
        the validated/coerced value or instance.
        """
        ...

    @abstractmethod
    def build_decision_model(self, spec: DecisionModelSpec) -> DecisionModel:
        """Build a decision model for a structured LLM response."""
        ...
