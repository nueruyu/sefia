from collections.abc import Iterator
from typing import cast

from sefia.llm.schema import JsonObject, SchemaNode, SchemaPath


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
    reference = SchemaNode(schema).reference
    if reference is None or not reference.startswith("#/$defs/"):
        return schema, path
    definitions = SchemaNode(root).object_map("$defs")
    if definitions is None:
        return schema, path
    name = reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
    resolved = definitions.get(name)
    if not isinstance(resolved, dict):
        return schema, path
    return resolved, ("$defs", name)


def matches(data: object, schema: JsonObject, root: JsonObject) -> bool:
    schema, _ = resolve(schema, root, ())
    node = SchemaNode(schema)
    if "const" in schema and data != schema["const"]:
        return False
    if node.type == "null":
        return data is None
    if node.type == "object":
        if not isinstance(data, dict):
            return False
        required = schema.get("required")
        required_names: set[str] = (
            {item for item in required if isinstance(item, str)}
            if isinstance(required, list)
            else set()
        )
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
