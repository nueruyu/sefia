from abc import ABC, abstractmethod
from typing import Any


class PromptFormatter(ABC):
    """Renders inference arguments as the complete task-argument user message."""

    @abstractmethod
    def format_arguments(self, arguments: dict[str, Any]) -> str:
        """Render a complete user message containing the prompt arguments."""
        ...
