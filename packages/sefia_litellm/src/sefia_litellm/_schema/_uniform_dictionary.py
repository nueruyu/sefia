from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import jsonschema.validators
from typing_extensions import final

from sefia.llm.json_schema import (
    JsonObject,
    JsonScalar,
    JsonValue,
    SchemaKeyword,
    SchemaNode,
    SchemaPath,
)
from sefia.llm.structured_data import StructuredData, StructuredDataTree

K = SchemaKeyword

_PRESERVED_ANNOTATIONS = (K.TITLE, K.DESCRIPTION)
_PROPERTY_TO_ITEM_CONSTRAINT = {
    K.MIN_PROPERTIES: K.MIN_ITEMS,
    K.MAX_PROPERTIES: K.MAX_ITEMS,
}


@final
@dataclass(frozen=True)
class UniformDictionarySchema:
    key_schema: JsonObject
    value_schema: JsonObject
    annotations: JsonObject
    entry_array_constraints: JsonObject

    @classmethod
    def from_node(cls, node: SchemaNode) -> "UniformDictionarySchema":
        values = node.additional_properties()
        if node.type != "object" or not isinstance(values, SchemaNode):
            raise ValueError("node is not a dictionary schema")
        if node.properties() or node.required():
            raise ValueError(
                "objects combining fixed properties with dictionary values "
                "cannot be lowered safely"
            )
        keys = node.property_names()
        return cls(
            key_schema=keys.value if keys is not None else {K.TYPE: "string"},
            value_schema=values.value,
            annotations={
                keyword: node.value[keyword]
                for keyword in _PRESERVED_ANNOTATIONS
                if keyword in node.value
            },
            entry_array_constraints={
                target: node.value[source]
                for source, target in _PROPERTY_TO_ITEM_CONSTRAINT.items()
                if source in node.value
            },
        )

    def to_entry_array_schema(self) -> JsonObject:
        entry = SchemaNode.object_schema(
            {"key": self.key_schema, "value": self.value_schema}
        )
        return {
            **self.annotations,
            **self.entry_array_constraints,
            K.TYPE: "array",
            K.ITEMS: entry.value,
        }


@final
@dataclass(frozen=True)
class UniformDictionaryFormat:
    schema: JsonObject
    mapping_paths: frozenset[SchemaPath]

    @classmethod
    def from_schema(cls, schema: JsonObject) -> "UniformDictionaryFormat":
        mapping_paths: set[SchemaPath] = set()
        while cursor := _find_dictionary_schema(schema):
            path, node, dictionary = cursor
            replacement = dictionary.to_entry_array_schema()
            node.value.clear()
            node.value.update(replacement)
            mapping_paths.add(path)
        return cls(schema, frozenset(mapping_paths))

    def decode(self, data: StructuredData) -> StructuredData:
        return _decode(data, self.schema, self.schema, self.mapping_paths, ())

    def encode(self, data: StructuredData) -> StructuredData:
        return _encode(data, self.schema, self.schema, self.mapping_paths, ())


def _find_dictionary_schema(
    schema: JsonObject,
) -> tuple[SchemaPath, SchemaNode, UniformDictionarySchema] | None:
    for cursor in SchemaNode(schema).walk():
        if _is_dictionary_schema(cursor.node):
            return (
                cursor.path,
                cursor.node,
                UniformDictionarySchema.from_node(cursor.node),
            )
    return None


def _is_dictionary_schema(node: SchemaNode) -> bool:
    return node.type == "object" and isinstance(
        node.additional_properties(), SchemaNode
    )


def _decode(
    data: StructuredData,
    schema: JsonObject,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> StructuredData:
    schema, path = _resolve(schema, root, path)
    node = SchemaNode(schema)
    if path in mapping_paths:
        return _decode_dictionary(data, node, root, mapping_paths, path)

    for index, alternative in enumerate(node.any_of()):
        if _matches(data.tree, alternative.value, root):
            return _decode(
                data,
                alternative.value,
                root,
                mapping_paths,
                (*path, K.ANY_OF, index),
            )

    if node.type == "object":
        return _decode_object(data, node, root, mapping_paths, path)
    if node.type == "array":
        return _decode_array(data, node, root, mapping_paths, path)
    return data


def _encode(
    data: StructuredData,
    schema: JsonObject,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> StructuredData:
    schema, path = _resolve(schema, root, path)
    node = SchemaNode(schema)
    if path in mapping_paths:
        return _encode_dictionary(data, node, root, mapping_paths, path)

    for index, alternative in enumerate(node.any_of()):
        candidate = _encode(
            data,
            alternative.value,
            root,
            mapping_paths,
            (*path, K.ANY_OF, index),
        )
        if _matches(candidate.tree, alternative.value, root):
            return candidate

    if node.type == "object":
        return _encode_object(data, node, root, mapping_paths, path)
    if node.type == "array":
        return _encode_array(data, node, root, mapping_paths, path)
    return data


def _resolve(
    schema: JsonObject,
    root: JsonObject,
    path: SchemaPath,
) -> tuple[JsonObject, SchemaPath]:
    node = SchemaNode(schema)
    reference = node.local_reference
    if reference is None:
        return schema, path
    resolved = node.resolve_local_reference(SchemaNode(root))
    if resolved is None:
        return schema, path
    return resolved.value, (K.DEFINITIONS, reference.definition, *reference.path)


def _matches(data: StructuredDataTree, schema: JsonObject, root: JsonObject) -> bool:
    candidate = deepcopy(schema)
    for keyword in (K.DEFINITIONS, K.LEGACY_DEFINITIONS):
        if keyword in root:
            candidate[keyword] = deepcopy(root[keyword])
    validator_cls = jsonschema.validators.validator_for(
        root, default=jsonschema.Draft202012Validator
    )
    return validator_cls(candidate).is_valid(cast(JsonValue, data))


def _decode_object(
    data: StructuredData,
    node: SchemaNode,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> StructuredData:
    try:
        fields = data.to_object()
    except ValueError:
        return data
    properties = node.properties()
    return StructuredData.from_object(
        {
            name: _decode(
                value,
                properties[name].value,
                root,
                mapping_paths,
                (*path, K.PROPERTIES, name),
            )
            if name in properties
            else value
            for name, value in fields.items()
        }
    )


def _decode_array(
    data: StructuredData,
    node: SchemaNode,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> StructuredData:
    items = node.items()
    if items is None:
        return data
    try:
        values = data.to_array()
    except ValueError:
        return data
    return StructuredData.from_array(
        _decode(value, items.value, root, mapping_paths, (*path, K.ITEMS))
        for value in values
    )


def _decode_dictionary(
    data: StructuredData,
    node: SchemaNode,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> StructuredData:
    items = node.items()
    properties = items.properties() if items is not None else {}
    if set(properties) != {"key", "value"}:
        raise ValueError("lowered mapping schema is missing key/value entries")
    result: dict[JsonScalar, StructuredData] = {}
    for entry in data.to_array():
        fields = entry.to_object("mapping entry")
        if set(fields) != {"key", "value"}:
            raise ValueError("mapping entries must contain only key and value")
        key = _decode(
            fields["key"],
            properties["key"].value,
            root,
            mapping_paths,
            (*path, K.ITEMS, K.PROPERTIES, "key"),
        ).to_scalar("mapping key")
        if key in result:
            raise ValueError(f"duplicate mapping key: {key!r}")
        result[key] = _decode(
            fields["value"],
            properties["value"].value,
            root,
            mapping_paths,
            (*path, K.ITEMS, K.PROPERTIES, "value"),
        )
    return StructuredData.from_mapping(result)


def _encode_dictionary(
    data: StructuredData,
    node: SchemaNode,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> StructuredData:
    values = data.tree
    if type(values) is not dict:
        return data
    items = node.items()
    properties = items.properties() if items is not None else {}
    if set(properties) != {"key", "value"}:
        raise ValueError("lowered mapping schema is missing key/value entries")
    return StructuredData.from_array(
        StructuredData.from_object(
            {
                "key": _encode(
                    StructuredData.from_scalar(key),
                    properties["key"].value,
                    root,
                    mapping_paths,
                    (*path, K.ITEMS, K.PROPERTIES, "key"),
                ),
                "value": _encode(
                    StructuredData.from_tree(value),
                    properties["value"].value,
                    root,
                    mapping_paths,
                    (*path, K.ITEMS, K.PROPERTIES, "value"),
                ),
            }
        )
        for key, value in values.items()
    )


def _encode_object(
    data: StructuredData,
    node: SchemaNode,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> StructuredData:
    try:
        fields = data.to_object()
    except ValueError:
        return data
    properties = node.properties()
    return StructuredData.from_object(
        {
            name: _encode(
                value,
                properties[name].value,
                root,
                mapping_paths,
                (*path, K.PROPERTIES, name),
            )
            if name in properties
            else value
            for name, value in fields.items()
        }
    )


def _encode_array(
    data: StructuredData,
    node: SchemaNode,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> StructuredData:
    items = node.items()
    if items is None:
        return data
    try:
        values = data.to_array()
    except ValueError:
        return data
    return StructuredData.from_array(
        _encode(value, items.value, root, mapping_paths, (*path, K.ITEMS))
        for value in values
    )
