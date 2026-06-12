from abc import ABC, abstractmethod

from .._tool_registry import ToolRegistry


class ToolCollector(ABC):
    @abstractmethod
    def collect(self, instance: object) -> ToolRegistry:
        ...
