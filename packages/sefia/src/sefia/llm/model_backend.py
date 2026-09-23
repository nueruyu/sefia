from abc import ABC, abstractmethod

from .._tool_system import ToolFunctionInspector
from .result_format import ResultFormatFactory
from .structured_data import StructuredData


class ModelBackend(ToolFunctionInspector, ResultFormatFactory, ABC):
    """Bridges Python models and Sefia's provider-neutral LLM data."""

    @abstractmethod
    def to_structured_data(self, value: object) -> StructuredData:
        """Normalize an application value for an LLM boundary."""


__all__ = ["ModelBackend"]
