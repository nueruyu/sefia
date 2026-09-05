from abc import ABC, abstractmethod
from typing import Any

from .json_schema import JsonSchemaDocument
from .structured_data import StructuredData


class ResultFormat(ABC):
    @property
    @abstractmethod
    def schema(self) -> JsonSchemaDocument: ...

    @abstractmethod
    def validate(self, data: StructuredData) -> Any: ...


class ResultFormatFactory(ABC):
    @abstractmethod
    def create(self, python_type: Any) -> ResultFormat: ...


__all__ = ["ResultFormat", "ResultFormatFactory"]
