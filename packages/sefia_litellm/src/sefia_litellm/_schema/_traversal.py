from collections.abc import Iterator
from typing import Any, cast

from sefia.llm.schema import SchemaPath

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


def walk(node: Any, *, skip: set[int] | None = None) -> Iterator[dict[str, Any]]:
    if isinstance(node, list):
        for item in cast(list[Any], node):
            yield from walk(item, skip=skip)
        return
    if not isinstance(node, dict):
        return
    schema = cast(dict[str, Any], node)
    if skip is not None and id(schema) in skip:
        return
    yield schema
    for _, child in _children(schema):
        yield from walk(child, skip=skip)


def walk_with_paths(
    node: Any, path: SchemaPath = ()
) -> Iterator[tuple[SchemaPath, dict[str, Any]]]:
    if isinstance(node, list):
        for index, item in enumerate(cast(list[Any], node)):
            yield from walk_with_paths(item, (*path, index))
        return
    if not isinstance(node, dict):
        return
    schema = cast(dict[str, Any], node)
    yield path, schema
    for child_path, child in _children(schema):
        yield from walk_with_paths(child, (*path, *child_path))


def resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
        return schema
    definitions = root.get("$defs")
    if not isinstance(definitions, dict):
        return schema
    name = reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
    resolved = cast(dict[str, Any], definitions).get(name)
    return cast(dict[str, Any], resolved) if isinstance(resolved, dict) else schema


def matches(data: Any, schema: dict[str, Any], root: dict[str, Any]) -> bool:
    schema = resolve(schema, root)
    if "const" in schema and data != schema["const"]:
        return False
    expected = schema.get("type")
    if expected == "null":
        return data is None
    if expected == "object":
        if not isinstance(data, dict):
            return False
        required = schema.get("required")
        return not isinstance(required, list) or set(cast(list[str], required)) <= set(
            cast(dict[str, Any], data)
        )
    if expected == "array":
        return isinstance(data, list)
    if expected == "string":
        return isinstance(data, str)
    if expected == "integer":
        return isinstance(data, int) and not isinstance(data, bool)
    if expected == "number":
        return isinstance(data, int | float) and not isinstance(data, bool)
    if expected == "boolean":
        return isinstance(data, bool)
    return True


def _children(node: dict[str, Any]) -> Iterator[tuple[tuple[str, ...], Any]]:
    for keyword in _MAP_CHILDREN:
        children = node.get(keyword)
        if isinstance(children, dict):
            for name, child in list(cast(dict[str, Any], children).items()):
                yield (keyword, name), child
    for keyword in _VALUE_CHILDREN:
        if keyword in node:
            yield (keyword,), node[keyword]
