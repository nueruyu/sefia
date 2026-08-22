import builtins
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from typing_extensions import final

from .json_schema import JsonScalar, JsonValue


@final
@dataclass(frozen=True)
class StructuredValue:
    _value: JsonScalar | list[builtins.object] | dict[builtins.object, builtins.object]

    @classmethod
    def from_json(cls, value: JsonValue) -> "StructuredValue":
        return cls._from_value(value)

    @classmethod
    def _from_value(cls, value: builtins.object) -> "StructuredValue":
        if isinstance(value, list):
            items = cast(list[builtins.object], value)
            return cls(list(items))
        if isinstance(value, dict):
            fields = cast(dict[builtins.object, builtins.object], value)
            return cls(dict(fields))
        if value is None or isinstance(value, str | int | float | bool):
            return cls(value)
        raise ValueError(f"Unsupported structured value: {value!r}")

    @classmethod
    def scalar(cls, value: JsonScalar) -> "StructuredValue":
        return cls(value)

    @classmethod
    def array(cls, values: Iterable["StructuredValue"]) -> "StructuredValue":
        return cls([value.value for value in values])

    @classmethod
    def object(cls, fields: Mapping[str, "StructuredValue"]) -> "StructuredValue":
        return cls({name: value.value for name, value in fields.items()})

    @classmethod
    def mapping(
        cls, entries: Mapping[JsonScalar, "StructuredValue"]
    ) -> "StructuredValue":
        return cls({key: value.value for key, value in entries.items()})

    @property
    def value(self) -> builtins.object:
        return self._value

    def to_object(self, description: str = "value") -> dict[str, "StructuredValue"]:
        if type(self._value) is not dict:
            raise ValueError(f"{description} must be an object")
        raw_fields = self._value
        if not all(isinstance(key, str) for key in raw_fields):
            raise ValueError(f"{description} must have string keys")
        return {
            key: StructuredValue._from_value(value)
            for key, value in raw_fields.items()
            if isinstance(key, str)
        }

    def to_array(self, description: str = "value") -> list["StructuredValue"]:
        if type(self._value) is not list:
            raise ValueError(f"{description} must be an array")
        return [StructuredValue._from_value(value) for value in self._value]

    def to_string(self, description: str = "value") -> str:
        if not isinstance(self._value, str):
            raise ValueError(f"{description} must be a string")
        return self._value

    def to_scalar(self, description: str = "value") -> JsonScalar:
        if self._value is None or isinstance(self._value, str | int | float | bool):
            return self._value
        raise ValueError(f"{description} must be a scalar")


__all__ = ["StructuredValue"]
