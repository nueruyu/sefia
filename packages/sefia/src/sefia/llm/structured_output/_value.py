from typing_extensions import TypeAlias

from ..schema import JsonScalar, JsonValue

StructuredValue: TypeAlias = (
    JsonScalar | list["StructuredValue"] | dict[JsonScalar, "StructuredValue"]
)


def to_structured_value(value: JsonValue) -> StructuredValue:
    if isinstance(value, list):
        return [to_structured_value(item) for item in value]
    if isinstance(value, dict):
        return {key: to_structured_value(item) for key, item in value.items()}
    return value
