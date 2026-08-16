from collections.abc import Iterator
from typing import cast

from sefia.llm.json_schema import JsonObject, SchemaKeyword, SchemaNode, SchemaPath

K = SchemaKeyword


def walk(
    root: JsonObject, *, skip: frozenset[SchemaPath] = frozenset()
) -> Iterator[tuple[SchemaPath, SchemaNode]]:
    for cursor in SchemaNode(root).walk():
        if any(cursor.path[: len(path)] == path for path in skip):
            continue
        yield cursor.path, cursor.node


def resolve(
    schema: JsonObject,
    root: JsonObject,
    path: SchemaPath,
) -> tuple[JsonObject, SchemaPath]:
    reference = SchemaNode(schema).local_reference
    if reference is None:
        return schema, path
    definitions = SchemaNode(root).object_map(K.DEFINITIONS)
    resolved = reference.resolve_from(definitions or {})
    if not isinstance(resolved, dict):
        return schema, path
    return resolved, (K.DEFINITIONS, reference.definition, *reference.path)


def matches(data: object, schema: JsonObject, root: JsonObject) -> bool:
    schema, _ = resolve(schema, root, ())
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
