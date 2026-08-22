from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from typing_extensions import final

from .json_schema import JsonScalar, JsonValue

LLMOutputData: TypeAlias = (
    JsonScalar | list["LLMOutputData"] | dict[JsonScalar, "LLMOutputData"]
)


@final
@dataclass(frozen=True)
class LLMOutput:
    _data: LLMOutputData

    @classmethod
    def from_json(cls, value: JsonValue) -> "LLMOutput":
        return cls._from_value(value)

    @classmethod
    def _from_value(cls, value: object) -> "LLMOutput":
        if isinstance(value, list):
            items = cast(list[LLMOutputData], value)
            return cls(list(items))
        if isinstance(value, dict):
            fields = cast(dict[JsonScalar, LLMOutputData], value)
            return cls(dict(fields))
        if value is None or isinstance(value, str | int | float | bool):
            return cls(value)
        raise ValueError(f"Unsupported LLM output: {value!r}")

    @classmethod
    def from_scalar(cls, value: JsonScalar) -> "LLMOutput":
        return cls(value)

    @classmethod
    def from_array(cls, values: Iterable["LLMOutput"]) -> "LLMOutput":
        return cls([value.data for value in values])

    @classmethod
    def from_object(cls, fields: Mapping[str, "LLMOutput"]) -> "LLMOutput":
        return cls({name: value.data for name, value in fields.items()})

    @classmethod
    def from_mapping(cls, entries: Mapping[JsonScalar, "LLMOutput"]) -> "LLMOutput":
        return cls({key: value.data for key, value in entries.items()})

    @property
    def data(self) -> LLMOutputData:
        return self._data

    def to_object(self, description: str = "output") -> dict[str, "LLMOutput"]:
        if type(self._data) is not dict:
            raise ValueError(f"{description} must be an object")
        raw_fields = self._data
        if not all(isinstance(key, str) for key in raw_fields):
            raise ValueError(f"{description} must have string keys")
        return {
            key: LLMOutput._from_value(value)
            for key, value in raw_fields.items()
            if isinstance(key, str)
        }

    def to_array(self, description: str = "output") -> list["LLMOutput"]:
        if type(self._data) is not list:
            raise ValueError(f"{description} must be an array")
        return [LLMOutput._from_value(value) for value in self._data]

    def to_string(self, description: str = "output") -> str:
        if not isinstance(self._data, str):
            raise ValueError(f"{description} must be a string")
        return self._data

    def to_scalar(self, description: str = "output") -> JsonScalar:
        if self._data is None or isinstance(self._data, str | int | float | bool):
            return self._data
        raise ValueError(f"{description} must be a scalar")


__all__ = ["LLMOutput", "LLMOutputData"]
