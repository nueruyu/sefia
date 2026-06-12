from abc import ABC, abstractmethod
from typing import Any


class PromptFormatter(ABC):
    """Formats inference arguments."""

    @abstractmethod
    def format_arguments(
        self, arguments: dict[str, Any], type_hints: dict[str, Any]
    ) -> str:
        """Return formatted arguments."""
        ...
