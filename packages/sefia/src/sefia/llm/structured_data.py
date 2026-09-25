import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from typing_extensions import final

from ..json_schema import JsonScalar, JsonValue

StructuredDataTree: TypeAlias = (
    JsonScalar | list["StructuredDataTree"] | dict[JsonScalar, "StructuredDataTree"]
)


class StructuredDataConverter(ABC):
    """Converts Python runtime values to provider-neutral structured data."""

    @abstractmethod
    def to_structured_data(self, value: object) -> "StructuredData": ...


@final
@dataclass(frozen=True)
class StructuredData:
    """Sefia's provider-neutral structured representation at LLM boundaries.

    It represents normalized application values supplied to an LLM and structured
    values decoded from an LLM or provider representation. Unlike JSON, logical
    mappings may retain scalar keys.
    """

    _tree: StructuredDataTree

    def __post_init__(self) -> None:
        object.__setattr__(self, "_tree", _copy_tree(self._tree))

    @classmethod
    def from_json(cls, value: JsonValue) -> "StructuredData":
        return cls(_copy_json_tree(value))

    @classmethod
    def parse_json(cls, text: str) -> "StructuredData":
        return cls(_copy_json_tree(cast(object, json.loads(text))))

    @classmethod
    def from_tree(cls, tree: StructuredDataTree) -> "StructuredData":
        return cls(tree)

    @classmethod
    def _from_tree(cls, tree: object) -> "StructuredData":
        return cls(cast(StructuredDataTree, tree))

    @classmethod
    def from_scalar(cls, value: JsonScalar) -> "StructuredData":
        return cls(value)

    @classmethod
    def from_array(cls, values: Iterable["StructuredData"]) -> "StructuredData":
        return cls([value._tree for value in values])

    @classmethod
    def from_object(cls, fields: Mapping[str, "StructuredData"]) -> "StructuredData":
        return cls({name: value._tree for name, value in fields.items()})

    @classmethod
    def from_mapping(
        cls, entries: Mapping[JsonScalar, "StructuredData"]
    ) -> "StructuredData":
        return cls({key: value._tree for key, value in entries.items()})

    @property
    def tree(self) -> StructuredDataTree:
        return _copy_tree(self._tree)

    def to_json_value(self) -> JsonValue:
        """Project this structured tree into JSON-compatible data."""
        return _to_json_value(self._tree)

    def to_object(
        self, description: str = "structured data"
    ) -> dict[str, "StructuredData"]:
        if type(self._tree) is not dict:
            raise ValueError(f"{description} must be an object")
        raw_fields = self._tree
        if not all(isinstance(key, str) for key in raw_fields):
            raise ValueError(f"{description} must have string keys")
        return {
            key: StructuredData._from_tree(value)
            for key, value in raw_fields.items()
            if isinstance(key, str)
        }

    def to_array(self, description: str = "structured data") -> list["StructuredData"]:
        if type(self._tree) is not list:
            raise ValueError(f"{description} must be an array")
        return [StructuredData._from_tree(value) for value in self._tree]

    def to_string(self, description: str = "structured data") -> str:
        if not isinstance(self._tree, str):
            raise ValueError(f"{description} must be a string")
        return self._tree

    def to_scalar(self, description: str = "structured data") -> JsonScalar:
        if self._tree is None or isinstance(self._tree, str | int | float | bool):
            return self._tree
        raise ValueError(f"{description} must be a scalar")


def _to_json_value(tree: StructuredDataTree) -> JsonValue:
    if isinstance(tree, list):
        return [_to_json_value(item) for item in tree]
    if isinstance(tree, dict):
        result: dict[str, JsonValue] = {}
        for key, value in tree.items():
            json_key = _to_json_key(key)
            if json_key in result:
                raise ValueError(
                    "Structured mapping contains keys that normalize to the same "
                    f"JSON key: {json_key!r}"
                )
            result[json_key] = _to_json_value(value)
        return result
    return tree


def _copy_tree(tree: object) -> StructuredDataTree:
    if isinstance(tree, list):
        return [_copy_tree(item) for item in cast(list[object], tree)]
    if isinstance(tree, dict):
        return {
            cast(JsonScalar, key): _copy_tree(value)
            for key, value in cast(dict[object, object], tree).items()
        }
    if tree is None or isinstance(tree, str | int | float | bool):
        return tree
    raise ValueError(f"Unsupported structured data: {tree!r}")


def _copy_json_tree(value: object) -> StructuredDataTree:
    if isinstance(value, list):
        return [_copy_json_tree(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        fields = cast(dict[object, object], value)
        if not all(isinstance(key, str) for key in fields):
            raise ValueError("JSON objects must have string keys")
        return {
            key: _copy_json_tree(item)
            for key, item in fields.items()
            if isinstance(key, str)
        }
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValueError(f"Unsupported JSON value: {value!r}")


def _to_json_key(key: JsonScalar) -> str:
    if key is None:
        return "null"
    if key is True:
        return "true"
    if key is False:
        return "false"
    return key if isinstance(key, str) else str(key)


__all__ = ["StructuredData", "StructuredDataConverter", "StructuredDataTree"]
