from abc import ABC, abstractmethod
from typing import Any


class PromptFormatter(ABC):
    """
    Abstract interface for formatting inference arguments for LLM prompts.
    """

    @abstractmethod
    def format_arguments(self, arguments: dict[str, Any]) -> str:
        """Serialize prompt arguments into a string."""
        ...
