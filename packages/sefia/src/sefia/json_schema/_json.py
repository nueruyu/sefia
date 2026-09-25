from collections.abc import Mapping
from typing import TypeGuard, cast

from typing_extensions import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def is_json_value(value: object) -> TypeGuard[JsonValue]:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(is_json_value(item) for item in cast(list[object], value))
    if isinstance(value, dict):
        items = cast(dict[object, object], value).items()
        return all(isinstance(key, str) and is_json_value(item) for key, item in items)
    return False


def require_json_object(value: Mapping[str, object] | object) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError("value is not a JSON object")
    candidate = cast(dict[object, object], value)
    if not all(
        isinstance(key, str) and is_json_value(item) for key, item in candidate.items()
    ):
        raise TypeError("value is not a JSON object")
    return cast(JsonObject, value)
