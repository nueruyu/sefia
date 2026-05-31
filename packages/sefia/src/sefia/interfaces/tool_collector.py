from abc import ABC, abstractmethod

from ..models import ToolRegistry


class ToolCollector(ABC):
    """Abstract base class for a tool collector."""

    @abstractmethod
    def collect(self, instance: object) -> ToolRegistry:
        """Collects tools from the given instance and returns a ToolRegistry."""
        ...
