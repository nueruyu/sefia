from abc import ABC, abstractmethod
from typing import Any

from .json_schema import JsonSchemaDocument
from .llm_output import LLMOutput


class ResultSchema(ABC):
    @property
    @abstractmethod
    def json_schema(self) -> JsonSchemaDocument: ...

    @abstractmethod
    def validate(self, value: LLMOutput) -> Any: ...


class ResultSchemaFactory(ABC):
    @abstractmethod
    def create(self, python_type: Any) -> ResultSchema: ...


__all__ = ["ResultSchema", "ResultSchemaFactory"]
