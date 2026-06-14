from abc import ABC, abstractmethod
from typing import Any, Callable, Type


class ModelInspector(ABC):
    """
    Interface for schema generation and model validation.
    """

    @abstractmethod
    def get_schema_for_type(self, model_type: Type[Any] | Any) -> dict:
        """Generate a JSON schema for a target type."""
        ...

    @abstractmethod
    def get_schema_for_function(self, func: Callable[..., Any]) -> dict:
        """Generate a tool-call schema for a function signature."""
        ...

    @abstractmethod
    def validate_and_create(self, model_type: Type[Any] | Any, data: Any) -> Any:
        """
        Validate arbitrary input data against a target type and return
        the validated/coerced value or instance.
        """
        ...
