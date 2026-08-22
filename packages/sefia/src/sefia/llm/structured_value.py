from dataclasses import dataclass
from collections.abc import Iterable
from typing import cast

from typing_extensions import TypeAlias, final

from .json_schema import JsonScalar, JsonValue

_StructuredData: TypeAlias = (
    JsonScalar | list["StructuredValue"] | dict[JsonScalar, "StructuredValue"]
)
StructuredPythonValue: TypeAlias = (
    JsonScalar
    | list["StructuredPythonValue"]
    | dict[JsonScalar, "StructuredPythonValue"]
)


@final
@dataclass(frozen=True)
class StructuredValue:
    _data: _StructuredData

    @classmethod
    def from_json(cls, value: JsonValue) -> "StructuredValue":
        if isinstance(value, list):
            return cls.array(cls.from_json(item) for item in value)
        if isinstance(value, dict):
            return cls.object({key: cls.from_json(item) for key, item in value.items()})
        return cls.scalar(value)

    @classmethod
    def scalar(cls, value: JsonScalar) -> "StructuredValue":
        return cls(value)

    @classmethod
    def array(cls, values: Iterable["StructuredValue"]) -> "StructuredValue":
        return cls(list(values))

    @classmethod
    def object(cls, fields: dict[JsonScalar, "StructuredValue"]) -> "StructuredValue":
        return cls(dict(fields))

    def as_object(
        self, description: str = "value"
    ) -> dict[JsonScalar, "StructuredValue"]:
        if not isinstance(self._data, dict):
            raise ValueError(f"{description} must be an object")
        return dict(self._data)

    def as_record(self, description: str = "value") -> dict[str, "StructuredValue"]:
        fields = self.as_object(description)
        if not all(isinstance(key, str) for key in fields):
            raise ValueError(f"{description} must have string keys")
        return cast(dict[str, StructuredValue], fields)

    def as_array(self, description: str = "value") -> list["StructuredValue"]:
        if not isinstance(self._data, list):
            raise ValueError(f"{description} must be an array")
        return list(self._data)

    def as_string(self, description: str = "value") -> str:
        if not isinstance(self._data, str):
            raise ValueError(f"{description} must be a string")
        return self._data

    def as_scalar(self, description: str = "value") -> JsonScalar:
        if isinstance(self._data, list | dict):
            raise ValueError(f"{description} must be a scalar")
        return self._data

    def to_python(self) -> StructuredPythonValue:
        if isinstance(self._data, list):
            return [item.to_python() for item in self._data]
        if isinstance(self._data, dict):
            return {key: item.to_python() for key, item in self._data.items()}
        return self._data


__all__ = ["StructuredPythonValue", "StructuredValue"]
