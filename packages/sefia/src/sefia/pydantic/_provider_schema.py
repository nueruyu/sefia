from dataclasses import dataclass
from typing import Any, cast

from typing_extensions import final


@final
@dataclass
class ProviderSchema:
    schema: dict[str, Any]
    _mapping_schema_ids: set[int]

    def decode(self, data: Any) -> Any:
        return _decode(data, self.schema, self.schema, self._mapping_schema_ids)


def lower_provider_schema(
    schema: dict[str, Any], *, preserved_schema_ids: set[int]
) -> ProviderSchema:
    """Lower typed schemas to strict-provider representations."""
    mapping_schema_ids: set[int] = set()

    def lower(node: Any) -> None:
        if isinstance(node, list):
            for item in cast(list[Any], node):
                lower(item)
            return
        if not isinstance(node, dict):
            return
        node_dict = cast(dict[str, Any], node)
        if id(node_dict) in preserved_schema_ids:
            return

        additional = node_dict.get("additionalProperties")
        if node_dict.get("type") == "object" and isinstance(additional, dict):
            value_schema = cast(dict[str, Any], additional)
            property_names = node_dict.get("propertyNames")
            key_schema = (
                cast(dict[str, Any], property_names)
                if isinstance(property_names, dict)
                else {"type": "string"}
            )
            metadata = {
                key: node_dict[key]
                for key in ("title", "description")
                if key in node_dict
            }
            if "minProperties" in node_dict:
                metadata["minItems"] = node_dict["minProperties"]
            if "maxProperties" in node_dict:
                metadata["maxItems"] = node_dict["maxProperties"]
            node_dict.clear()
            node_dict.update(
                metadata,
                type="array",
                items={
                    "type": "object",
                    "properties": {
                        "key": key_schema,
                        "value": value_schema,
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            )
            mapping_schema_ids.add(id(node_dict))

        for child in _schema_children(node_dict):
            lower(child)

    lower(schema)
    return ProviderSchema(schema=schema, _mapping_schema_ids=mapping_schema_ids)


def _decode(
    data: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    mapping_schema_ids: set[int],
) -> Any:
    schema = _resolve(schema, root)

    if id(schema) in mapping_schema_ids:
        if not isinstance(data, list):
            return data
        item_schema = cast(dict[str, Any], schema["items"])
        properties = cast(dict[str, Any], item_schema["properties"])
        key_schema = cast(dict[str, Any], properties["key"])
        value_schema = cast(dict[str, Any], properties["value"])
        mapping_result: dict[Any, Any] = {}
        for entry in cast(list[Any], data):
            if not isinstance(entry, dict):
                raise ValueError("mapping entries must be objects")
            entry_dict = cast(dict[str, Any], entry)
            if set(entry_dict) != {"key", "value"}:
                raise ValueError("mapping entries must contain only key and value")
            key = _decode(entry_dict["key"], key_schema, root, mapping_schema_ids)
            value = _decode(entry_dict["value"], value_schema, root, mapping_schema_ids)
            try:
                if key in mapping_result:
                    raise ValueError(f"duplicate mapping key: {key!r}")
                mapping_result[key] = value
            except TypeError as error:
                raise ValueError("mapping keys must be hashable") from error
        return mapping_result

    candidates = schema.get("anyOf")
    if isinstance(candidates, list):
        for candidate in cast(list[Any], candidates):
            if not isinstance(candidate, dict):
                continue
            candidate_schema = cast(dict[str, Any], candidate)
            if _matches(data, candidate_schema, root):
                return _decode(data, candidate_schema, root, mapping_schema_ids)
        return data

    if schema.get("type") == "object" and isinstance(data, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return cast(dict[str, Any], data)
        property_schemas = cast(dict[str, Any], properties)
        object_result: dict[str, Any] = {
            key: _decode(value, property_schemas[key], root, mapping_schema_ids)
            if key in property_schemas
            else value
            for key, value in cast(dict[str, Any], data).items()
        }
        return object_result

    if schema.get("type") == "array" and isinstance(data, list):
        items = schema.get("items")
        if isinstance(items, dict):
            item_schema = cast(dict[str, Any], items)
            array_result: list[Any] = [
                _decode(item, item_schema, root, mapping_schema_ids)
                for item in cast(list[Any], data)
            ]
            return array_result

    return cast(Any, data)


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
        return schema
    name = reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
    definitions = root.get("$defs")
    if not isinstance(definitions, dict):
        return schema
    resolved = cast(dict[str, Any], definitions).get(name)
    return cast(dict[str, Any], resolved) if isinstance(resolved, dict) else schema


def _matches(data: Any, schema: dict[str, Any], root: dict[str, Any]) -> bool:
    schema = _resolve(schema, root)
    if "const" in schema and data != schema["const"]:
        return False
    schema_type = schema.get("type")
    if schema_type == "null":
        return data is None
    if schema_type == "object":
        if not isinstance(data, dict):
            return False
        required = schema.get("required")
        return not isinstance(required, list) or set(cast(list[str], required)) <= set(
            cast(dict[str, Any], data)
        )
    if schema_type == "array":
        return isinstance(data, list)
    if schema_type == "string":
        return isinstance(data, str)
    if schema_type == "integer":
        return isinstance(data, int) and not isinstance(data, bool)
    if schema_type == "number":
        return isinstance(data, int | float) and not isinstance(data, bool)
    if schema_type == "boolean":
        return isinstance(data, bool)
    return True


def _schema_children(node: dict[str, Any]) -> list[Any]:
    children: list[Any] = []
    for keyword in ("$defs", "definitions", "properties", "patternProperties"):
        value = node.get(keyword)
        if isinstance(value, dict):
            children.extend(cast(dict[str, Any], value).values())
    for keyword in (
        "additionalProperties",
        "anyOf",
        "items",
        "oneOf",
        "prefixItems",
        "propertyNames",
    ):
        if keyword in node:
            children.append(node[keyword])
    return children
