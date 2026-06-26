from abc import ABC, abstractmethod
from typing import Any, Callable, Type


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
    def get_arguments_model(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
    ) -> Type[Any]:
        """
        Build a model representing a function's call arguments.

        The model can be used both to generate a strict JSON schema for the
        arguments and to validate/coerce a raw arguments mapping. Unknown
        arguments are rejected.
        """
        ...

    @abstractmethod
    def validate(self, model_type: Type[Any] | Any, data: Any) -> Any:
        """
        Validate arbitrary input data against a target type and return
        the validated/coerced value or instance.
        """
        ...
