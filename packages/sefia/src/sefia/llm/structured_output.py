from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import TypeAlias

from .json_schema import JsonScalar, JsonSchemaDocument, JsonValue

StructuredValue: TypeAlias = (
    JsonScalar | list["StructuredValue"] | dict[JsonScalar, "StructuredValue"]
)


def to_structured_value(value: JsonValue) -> StructuredValue:
    if isinstance(value, list):
        return [to_structured_value(item) for item in value]
    if isinstance(value, dict):
        return {key: to_structured_value(item) for key, item in value.items()}
    return value


class StructuredValueSchema(ABC):
    @property
    @abstractmethod
    def json_schema(self) -> JsonSchemaDocument: ...

    @abstractmethod
    def validate(self, value: StructuredValue) -> Any: ...


class StructuredValueSchemaFactory(ABC):
    @abstractmethod
    def create(self, python_type: Any) -> StructuredValueSchema: ...


__all__ = [
    "StructuredValue",
    "StructuredValueSchema",
    "StructuredValueSchemaFactory",
    "to_structured_value",
]
