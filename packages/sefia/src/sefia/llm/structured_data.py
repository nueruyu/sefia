import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from typing_extensions import final

from .json_schema import JsonScalar, JsonValue

StructuredDataTree: TypeAlias = (
    JsonScalar | list["StructuredDataTree"] | dict[JsonScalar, "StructuredDataTree"]
)


@final
@dataclass(frozen=True)
class StructuredData:
    """A provider-neutral data tree used by LLM clients and transports.

    Unlike JSON, logical mappings may retain scalar keys after an adapter restores
    a provider-specific wire representation.
    """

    _tree: StructuredDataTree

    @classmethod
    def from_json(cls, value: JsonValue) -> "StructuredData":
        return cls._from_json_value(value)

    @classmethod
    def parse_json(cls, text: str) -> "StructuredData":
        return cls._from_json_value(cast(object, json.loads(text)))

    @classmethod
    def from_tree(cls, tree: StructuredDataTree) -> "StructuredData":
        return cls._from_tree(tree)

    @classmethod
    def _from_json_value(cls, value: object) -> "StructuredData":
        if isinstance(value, list):
            items = cast(list[object], value)
            return cls([cls._from_json_value(item).tree for item in items])
        if isinstance(value, dict):
            fields = cast(dict[object, object], value)
            if not all(isinstance(key, str) for key in fields):
                raise ValueError("JSON objects must have string keys")
            return cls(
                {
                    key: cls._from_json_value(item).tree
                    for key, item in fields.items()
                    if isinstance(key, str)
                }
            )
        if value is None or isinstance(value, str | int | float | bool):
            return cls(value)
        raise ValueError(f"Unsupported JSON value: {value!r}")

    @classmethod
    def _from_tree(cls, tree: object) -> "StructuredData":
        if isinstance(tree, list):
            items = cast(list[StructuredDataTree], tree)
            return cls(list(items))
        if isinstance(tree, dict):
            fields = cast(dict[JsonScalar, StructuredDataTree], tree)
            return cls(dict(fields))
        if tree is None or isinstance(tree, str | int | float | bool):
            return cls(tree)
        raise ValueError(f"Unsupported structured data: {tree!r}")

    @classmethod
    def from_scalar(cls, value: JsonScalar) -> "StructuredData":
        return cls(value)

    @classmethod
    def from_array(cls, values: Iterable["StructuredData"]) -> "StructuredData":
        return cls([value.tree for value in values])

    @classmethod
    def from_object(cls, fields: Mapping[str, "StructuredData"]) -> "StructuredData":
        return cls({name: value.tree for name, value in fields.items()})

    @classmethod
    def from_mapping(
        cls, entries: Mapping[JsonScalar, "StructuredData"]
    ) -> "StructuredData":
        return cls({key: value.tree for key, value in entries.items()})

    @property
    def tree(self) -> StructuredDataTree:
        return self._tree

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


__all__ = ["StructuredData", "StructuredDataTree"]
