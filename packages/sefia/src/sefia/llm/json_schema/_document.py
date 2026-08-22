from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, cast

from typing_extensions import TypeAlias, final

from ._json import JsonObject, JsonValue, require_json_object
from ._path import SchemaPath
from ._reference import LocalDefinitionRef
from ._vocabulary import SchemaKeyword

K = SchemaKeyword

SchemaType: TypeAlias = Literal[
    "null", "boolean", "integer", "number", "string", "array", "object"
]

_MAP_CHILDREN = (
    K.DEFINITIONS,
    K.LEGACY_DEFINITIONS,
    K.PROPERTIES,
    K.PATTERN_PROPERTIES,
)
_VALUE_CHILDREN = (
    K.ADDITIONAL_PROPERTIES,
    K.ANY_OF,
    K.ALL_OF,
    K.CONTAINS,
    K.CONTENT_SCHEMA,
    K.DEPENDENT_SCHEMAS,
    K.ELSE,
    K.IF,
    K.ITEMS,
    K.NOT,
    K.ONE_OF,
    K.PREFIX_ITEMS,
    K.PROPERTY_NAMES,
    K.THEN,
    K.UNEVALUATED_ITEMS,
    K.UNEVALUATED_PROPERTIES,
)


@final
@dataclass
class SchemaNode:
    value: JsonObject

    @classmethod
    def object_schema(
        cls,
        properties: JsonObject,
        *,
        required: tuple[str, ...] | None = None,
        closed: bool = True,
    ) -> "SchemaNode":
        value: JsonObject = {
            K.TYPE: "object",
            K.PROPERTIES: properties,
            K.REQUIRED: list(required if required is not None else properties),
        }
        if closed:
            value[K.ADDITIONAL_PROPERTIES] = False
        return cls(value)

    @property
    def type(self) -> SchemaType | None:
        value = self.value.get(K.TYPE)
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
        value = self.value.get(K.REFERENCE)
        return value if isinstance(value, str) else None

    @property
    def local_reference(self) -> LocalDefinitionRef | None:
        value = self.value.get(K.REFERENCE)
        return LocalDefinitionRef.parse(value) if isinstance(value, str) else None

    def set_local_reference(self, reference: LocalDefinitionRef) -> None:
        self.value[K.REFERENCE] = reference.render()

    def resolve_local_reference(self, root: "SchemaNode") -> "SchemaNode | None":
        reference = self.local_reference
        if reference is None:
            return None
        definitions = root.object_map(K.DEFINITIONS)
        resolved = reference.resolve_from(definitions or {})
        return SchemaNode(resolved) if isinstance(resolved, dict) else None

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
        values = self.object_map(K.PROPERTIES)
        if values is None:
            return {}
        return {
            name: SchemaNode(value)
            for name, value in values.items()
            if isinstance(value, dict)
        }

    def definitions(self) -> dict[str, "SchemaNode"]:
        values = self.object_map(K.DEFINITIONS)
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

    def any_of(self) -> list["SchemaNode"]:
        return self.nodes(K.ANY_OF)

    def one_of(self) -> list["SchemaNode"]:
        return self.nodes(K.ONE_OF)

    def required(self) -> tuple[str, ...] | None:
        return self.strings(K.REQUIRED)

    def additional_properties(self) -> "bool | SchemaNode | None":
        value = self.value.get(K.ADDITIONAL_PROPERTIES)
        if isinstance(value, bool):
            return value
        return SchemaNode(value) if isinstance(value, dict) else None

    def items(self) -> "SchemaNode | None":
        return self.child(K.ITEMS)

    def property_names(self) -> "SchemaNode | None":
        return self.child(K.PROPERTY_NAMES)

    def ensure_definitions(self) -> JsonObject:
        definitions = self.value.setdefault(K.DEFINITIONS, {})
        if not isinstance(definitions, dict):
            raise ValueError("JSON Schema $defs must be an object")
        return definitions

    def set_definitions(self, definitions: JsonObject) -> None:
        self.value[K.DEFINITIONS] = definitions

    def take_definitions(self) -> JsonObject:
        definitions: JsonObject = {}
        for keyword in (K.DEFINITIONS, K.LEGACY_DEFINITIONS):
            value = self.value.pop(keyword, None)
            if isinstance(value, dict):
                definitions.update(value)
        return definitions

    def set_description(self, description: str) -> None:
        self.value[K.DESCRIPTION] = description

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
