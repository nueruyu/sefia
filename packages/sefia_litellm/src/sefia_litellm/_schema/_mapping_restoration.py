from dataclasses import dataclass
from typing import cast

from typing_extensions import final

from sefia.llm.json_schema import (
    JsonObject,
    JsonScalar,
    SchemaKeyword,
    SchemaNode,
    SchemaPath,
)
from sefia.llm.llm_output import LLMOutput

K = SchemaKeyword


@final
@dataclass(frozen=True)
class MappingEntry:
    key: LLMOutput
    value: LLMOutput

    @classmethod
    def from_output(cls, output: LLMOutput) -> "MappingEntry":
        fields = output.to_object("mapping entry")
        if set(fields) != {"key", "value"}:
            raise ValueError("mapping entries must contain only key and value")
        return cls(fields["key"], fields["value"])


def restore_mappings(
    output: LLMOutput,
    *,
    schema: JsonObject,
    restoration_paths: frozenset[SchemaPath],
) -> LLMOutput:
    return _restore(output, schema, schema, restoration_paths, ())


def _restore(
    output: LLMOutput,
    schema: JsonObject,
    root: JsonObject,
    restoration_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> LLMOutput:
    schema, path = _resolve(schema, root, path)
    node = SchemaNode(schema)
    if path in restoration_paths:
        return _restore_mapping(output, node, root, restoration_paths, path)

    alternatives = node.any_of()
    if alternatives:
        for index, alternative in enumerate(alternatives):
            if _matches(output.data, alternative.value, root):
                return _restore(
                    output,
                    alternative.value,
                    root,
                    restoration_paths,
                    (*path, K.ANY_OF, index),
                )
        return output

    if node.type == "object":
        return _restore_object(output, node, root, restoration_paths, path)
    if node.type == "array":
        return _restore_array(output, node, root, restoration_paths, path)
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
    restoration_paths: frozenset[SchemaPath],
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
                restoration_paths,
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
    restoration_paths: frozenset[SchemaPath],
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
        _restore(value, items.value, root, restoration_paths, (*path, K.ITEMS))
        for value in values
    )


def _restore_mapping(
    output: LLMOutput,
    node: SchemaNode,
    root: JsonObject,
    restoration_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> LLMOutput:
    items = node.items()
    properties = items.properties() if items is not None else {}
    if set(properties) != {"key", "value"}:
        raise ValueError("lowered mapping schema is missing key/value entries")
    entries = output.to_array()
    result: dict[JsonScalar, LLMOutput] = {}
    for output_entry in entries:
        entry = MappingEntry.from_output(output_entry)
        key = _restore(
            entry.key,
            properties["key"].value,
            root,
            restoration_paths,
            (*path, K.ITEMS, K.PROPERTIES, "key"),
        ).to_scalar("mapping key")
        if key in result:
            raise ValueError(f"duplicate mapping key: {key!r}")
        result[key] = _restore(
            entry.value,
            properties["value"].value,
            root,
            restoration_paths,
            (*path, K.ITEMS, K.PROPERTIES, "value"),
        )
    return LLMOutput.from_mapping(result)
