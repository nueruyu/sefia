from abc import ABC, abstractmethod
from typing import Any


class PromptFormatter(ABC):
    """
    Abstract interface for a strategy that formats inference arguments for inclusion
    in LLM prompts.
    """

    @abstractmethod
    def format_arguments(
        self, arguments: dict[str, Any], type_hints: dict[str, Any]
    ) -> str:
        """Serialize prompt arguments into a string."""
        ...
