from dataclasses import dataclass
from typing import cast

from typing_extensions import final

from sefia.llm.json_schema import (
    JsonObject,
    JsonScalar,
    JsonSchemaDocument,
    SchemaKeyword,
    SchemaNode,
    SchemaPath,
)
from sefia.llm.llm_output import LLMOutput

K = SchemaKeyword


@final
@dataclass(frozen=True)
class MappingTransform:
    schema: JsonObject
    mapping_paths: frozenset[SchemaPath]

    @classmethod
    def lower(cls, schema: JsonObject) -> "MappingTransform":
        mapping_paths: set[SchemaPath] = set()
        for cursor in list(SchemaNode(schema).walk()):
            path, node = cursor.path, cursor.node
            additional = node.additional_properties()
            if node.type != "object" or not isinstance(additional, SchemaNode):
                continue
            property_names = node.property_names()
            key_schema: JsonObject = (
                property_names.value
                if property_names is not None
                else {K.TYPE: "string"}
            )
            _replace_with_entries(node, key_schema, additional.value)
            mapping_paths.add(path)
        restoration_schema = JsonSchemaDocument(schema).mutable_copy()
        return cls(restoration_schema, frozenset(mapping_paths))

    def restore(self, output: LLMOutput) -> LLMOutput:
        return _restore(output, self.schema, self.schema, self.mapping_paths, ())


def _replace_with_entries(
    node: SchemaNode, key_schema: JsonObject, value_schema: JsonObject
) -> None:
    replacement: JsonObject = {
        keyword: node.value[keyword]
        for keyword in (K.TITLE, K.DESCRIPTION)
        if keyword in node.value
    }
    for source, target in (
        (K.MIN_PROPERTIES, K.MIN_ITEMS),
        (K.MAX_PROPERTIES, K.MAX_ITEMS),
    ):
        if source in node.value:
            replacement[target] = node.value[source]
    replacement.update(
        {
            K.TYPE: "array",
            K.ITEMS: {
                K.TYPE: "object",
                K.PROPERTIES: {"key": key_schema, "value": value_schema},
                K.REQUIRED: ["key", "value"],
                K.ADDITIONAL_PROPERTIES: False,
            },
        }
    )
    node.value.clear()
    node.value.update(replacement)


def _restore(
    output: LLMOutput,
    schema: JsonObject,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> LLMOutput:
    schema, path = _resolve(schema, root, path)
    node = SchemaNode(schema)
    if path in mapping_paths:
        return _restore_mapping(output, node, root, mapping_paths, path)

    alternatives = node.any_of()
    if alternatives:
        for index, alternative in enumerate(alternatives):
            if _matches(output.data, alternative.value, root):
                return _restore(
                    output,
                    alternative.value,
                    root,
                    mapping_paths,
                    (*path, K.ANY_OF, index),
                )
        return output

    if node.type == "object":
        return _restore_object(output, node, root, mapping_paths, path)
    if node.type == "array":
        return _restore_array(output, node, root, mapping_paths, path)
    return output


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


def _matches(data: object, schema: JsonObject, root: JsonObject) -> bool:
    schema, _ = _resolve(schema, root, ())
    node = SchemaNode(schema)
    if K.CONST in schema and data != schema[K.CONST]:
        return False
    if node.type == "null":
        return data is None
    if node.type == "object":
        if not isinstance(data, dict):
            return False
        required_names = set(node.required() or ())
        return required_names <= set(cast(dict[object, object], data).keys())
    if node.type == "array":
        return isinstance(data, list)
    if node.type == "string":
        return isinstance(data, str)
    if node.type == "integer":
        return isinstance(data, int) and not isinstance(data, bool)
    if node.type == "number":
        return isinstance(data, int | float) and not isinstance(data, bool)
    if node.type == "boolean":
        return isinstance(data, bool)
    return True


def _restore_object(
    output: LLMOutput,
    node: SchemaNode,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> LLMOutput:
    try:
        fields = output.to_object()
    except ValueError:
        return output
    properties = node.properties()
    return LLMOutput.from_object(
        {
            name: _restore(
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


def _restore_array(
    output: LLMOutput,
    node: SchemaNode,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> LLMOutput:
    items = node.items()
    if items is None:
        return output
    try:
        values = output.to_array()
    except ValueError:
        return output
    return LLMOutput.from_array(
        _restore(value, items.value, root, mapping_paths, (*path, K.ITEMS))
        for value in values
    )


def _restore_mapping(
    output: LLMOutput,
    node: SchemaNode,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> LLMOutput:
    items = node.items()
    properties = items.properties() if items is not None else {}
    if set(properties) != {"key", "value"}:
        raise ValueError("lowered mapping schema is missing key/value entries")
    entries = output.to_array()
    result: dict[JsonScalar, LLMOutput] = {}
    for entry in entries:
        fields = entry.to_object("mapping entry")
        if set(fields) != {"key", "value"}:
            raise ValueError("mapping entries must contain only key and value")
        key = _restore(
            fields["key"],
            properties["key"].value,
            root,
            mapping_paths,
            (*path, K.ITEMS, K.PROPERTIES, "key"),
        ).to_scalar("mapping key")
        if key in result:
            raise ValueError(f"duplicate mapping key: {key!r}")
        result[key] = _restore(
            fields["value"],
            properties["value"].value,
            root,
            mapping_paths,
            (*path, K.ITEMS, K.PROPERTIES, "value"),
        )
    return LLMOutput.from_mapping(result)
