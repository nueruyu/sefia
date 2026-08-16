from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, cast

from typing_extensions import TypeAlias, final

from ._json import JsonObject, JsonValue, require_json_object
from ._path import SchemaPath

SchemaType: TypeAlias = Literal[
    "null", "boolean", "integer", "number", "string", "array", "object"
]

_MAP_CHILDREN = ("$defs", "definitions", "properties", "patternProperties")
_VALUE_CHILDREN = (
    "additionalProperties",
    "anyOf",
    "allOf",
    "contains",
    "contentSchema",
    "dependentSchemas",
    "else",
    "if",
    "items",
    "not",
    "oneOf",
    "prefixItems",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
)


@final
@dataclass
class SchemaNode:
    value: JsonObject

    @property
    def type(self) -> SchemaType | None:
        value = self.value.get("type")
        if value in {
            "null",
            "boolean",
            "integer",
            "number",
            "string",
            "array",
            "object",
        }:
            return cast(SchemaType, value)
        return None

    @property
    def reference(self) -> str | None:
        value = self.value.get("$ref")
        return value if isinstance(value, str) else None

    def object_map(self, keyword: str) -> dict[str, JsonValue] | None:
        value = self.value.get(keyword)
        return value if isinstance(value, dict) else None

    def string(self, keyword: str) -> str | None:
        value = self.value.get(keyword)
        return value if isinstance(value, str) else None

    def strings(self, keyword: str) -> tuple[str, ...] | None:
        value = self.value.get(keyword)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            return None
        return cast(tuple[str, ...], tuple(value))

    def nodes(self, keyword: str) -> list["SchemaNode"]:
        value = self.value.get(keyword)
        if not isinstance(value, list):
            return []
        return [SchemaNode(item) for item in value if isinstance(item, dict)]

    def properties(self) -> dict[str, "SchemaNode"]:
        values = self.object_map("properties")
        if values is None:
            return {}
        return {
            name: SchemaNode(value)
            for name, value in values.items()
            if isinstance(value, dict)
        }

    def definitions(self) -> dict[str, "SchemaNode"]:
        values = self.object_map("$defs")
        if values is None:
            return {}
        return {
            name: SchemaNode(value)
            for name, value in values.items()
            if isinstance(value, dict)
        }

    def child(self, keyword: str) -> "SchemaNode | None":
        value = self.value.get(keyword)
        return SchemaNode(value) if isinstance(value, dict) else None

    def alternatives(self, keyword: Literal["anyOf", "oneOf"]) -> list["SchemaNode"]:
        return self.nodes(keyword)

    def additional_properties(self) -> "bool | SchemaNode | None":
        value = self.value.get("additionalProperties")
        if isinstance(value, bool):
            return value
        return SchemaNode(value) if isinstance(value, dict) else None

    def walk(self) -> Iterator["SchemaCursor"]:
        yield from _walk(self.value)


@final
@dataclass(frozen=True)
class SchemaCursor:
    path: SchemaPath
    node: SchemaNode


@final
class JsonSchemaDocument:
    def __init__(self, root: JsonObject):
        self._root = deepcopy(root)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | object) -> "JsonSchemaDocument":
        return cls(require_json_object(value))

    def to_dict(self) -> JsonObject:
        return deepcopy(self._root)

    def mutable_copy(self) -> JsonObject:
        return self.to_dict()

    def root(self) -> SchemaNode:
        return SchemaNode(self._root)

    def walk(self) -> Iterator[SchemaCursor]:
        yield from _walk(self._root)


def _walk(value: JsonValue, path: SchemaPath = ()) -> Iterator[SchemaCursor]:
    if not isinstance(value, dict):
        return
    node = SchemaNode(value)
    yield SchemaCursor(path, node)
    for keyword in _MAP_CHILDREN:
        children = node.object_map(keyword)
        if children is not None:
            for name, child in children.items():
                yield from _walk(child, (*path, keyword, name))
    for keyword in _VALUE_CHILDREN:
        child = value.get(keyword)
        if isinstance(child, list):
            for index, item in enumerate(child):
                yield from _walk(item, (*path, keyword, index))
        elif child is not None:
            yield from _walk(child, (*path, keyword))
