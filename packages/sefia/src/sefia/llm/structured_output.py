from dataclasses import dataclass

from typing_extensions import TypeAlias, final

from .json_schema import JsonScalar, JsonSchemaDocument, JsonValue, SchemaPath

StructuredValue: TypeAlias = (
    JsonScalar | list["StructuredValue"] | dict[JsonScalar, "StructuredValue"]
)


def to_structured_value(value: JsonValue) -> StructuredValue:
    if isinstance(value, list):
        return [to_structured_value(item) for item in value]
    if isinstance(value, dict):
        return {key: to_structured_value(item) for key, item in value.items()}
    return value


@final
@dataclass(frozen=True)
class StructuredOutputSchema:
    document: JsonSchemaDocument
    preserved_schema_paths: frozenset[SchemaPath] = frozenset()


__all__ = ["StructuredOutputSchema", "StructuredValue", "to_structured_value"]
