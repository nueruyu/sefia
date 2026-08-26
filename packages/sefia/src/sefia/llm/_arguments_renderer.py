from abc import ABC, abstractmethod
from typing import Any


class ArgumentsRenderer(ABC):
    """Renders inference arguments as a task-argument user message."""

    @abstractmethod
    def render(self, arguments: dict[str, Any]) -> str:
        """Render the complete content of the task-argument user message."""
        ...
