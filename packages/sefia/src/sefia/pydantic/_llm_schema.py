from copy import deepcopy
from collections.abc import Iterator
from typing import Any, cast

from pydantic import ConfigDict, TypeAdapter, create_model

_UNSUPPORTED_COMPOSITION_KEYWORDS = (
    "allOf",
    "not",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
)
_SCHEMA_MAP_KEYWORDS = ("$defs", "definitions", "properties", "patternProperties")
_SCHEMA_VALUE_KEYWORDS = (
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


def build_llm_schema(
    model: Any,
    *,
    name: str,
    raw_tool_schemas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the provider-facing envelope around a validation model."""
    envelope = create_model(
        f"{name}Envelope",
        __config__=ConfigDict(extra="forbid"),
        payload=(model, ...),
    )
    schema = _normalize_schema(
        TypeAdapter(envelope).json_schema(), raw_tool_schemas=raw_tool_schemas
    )
    _validate_provider_schema(schema)
    schema["description"] = "The model for the LLM's decision on the next action."
    return schema


def _normalize_schema(
    schema: dict[str, Any], *, raw_tool_schemas: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Return a closed schema accepted by the supported live providers."""
    result = deepcopy(schema)
    preserved_ids = _restore_raw_argument_schemas(result, raw_tool_schemas)

    def normalize(node: Any) -> None:
        if isinstance(node, list):
            for item in cast(list[Any], node):
                normalize(item)
            return
        if not isinstance(node, dict):
            return
        node_dict = cast(dict[str, Any], node)
        if id(node_dict) in preserved_ids:
            return

        if node_dict.get("type") == "object":
            node_dict.setdefault("additionalProperties", False)
            properties = node_dict.get("properties")
            if isinstance(properties, dict):
                node_dict["required"] = list(cast(dict[str, Any], properties))

        # OpenAI accepts nested ``anyOf`` but rejects ``oneOf``. The validation
        # model remains unchanged and enforces the original union semantics.
        one_of = node_dict.pop("oneOf", None)
        if one_of is not None:
            node_dict["anyOf"] = one_of

        for child in _schema_children(node_dict):
            normalize(child)

    normalize(result)
    return result


def _restore_raw_argument_schemas(
    schema: dict[str, Any], raw_tool_schemas: dict[str, dict[str, Any]]
) -> set[int]:
    preserved: set[int] = set()

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in cast(list[Any], node):
                visit(item)
            return
        if not isinstance(node, dict):
            return
        node_dict = cast(dict[str, Any], node)
        if id(node_dict) in preserved:
            return
        properties = node_dict.get("properties")
        if isinstance(properties, dict):
            property_dict = cast(dict[str, Any], properties)
            name_schema = property_dict.get("name")
            arguments_schema = property_dict.get("arguments")
            if (
                isinstance(name_schema, dict)
                and (tool_name := cast(dict[str, Any], name_schema).get("const"))
                in raw_tool_schemas
                and isinstance(arguments_schema, dict)
            ):
                restored = deepcopy(raw_tool_schemas[cast(str, tool_name)])
                property_dict["arguments"] = restored
                preserved.add(id(restored))
        for child in _schema_children(node_dict):
            visit(child)

    visit(schema)
    return preserved


def _validate_provider_schema(schema: dict[str, Any]) -> None:
    def validate(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, list):
            for index, item in enumerate(cast(list[Any], node)):
                validate(item, (*path, str(index)))
            return
        if not isinstance(node, dict):
            return
        node_dict = cast(dict[str, Any], node)

        if "oneOf" in node_dict:
            _unsupported(path, "oneOf is not supported; use a disjoint anyOf")
        for keyword in _UNSUPPORTED_COMPOSITION_KEYWORDS:
            if keyword in node_dict:
                _unsupported(path, f"{keyword} is not supported")

        if node_dict.get("type") == "object":
            if node_dict.get("additionalProperties") is not False:
                _unsupported(
                    path, "object schemas must set additionalProperties to false"
                )
            properties = node_dict.get("properties")
            if isinstance(properties, dict):
                property_names = set(cast(dict[str, Any], properties))
                required = node_dict.get("required")
                required_names: set[str] = (
                    set(cast(list[str], required))
                    if isinstance(required, list)
                    else set()
                )
                missing = sorted(property_names - required_names)
                if missing:
                    _unsupported(
                        path,
                        f"all object properties must be required; missing {missing}",
                    )

        for key, child in _schema_children_with_paths(node_dict):
            validate(child, (*path, *key))

    validate(schema, ())


def _schema_children(node: dict[str, Any]) -> Iterator[Any]:
    for _, child in _schema_children_with_paths(node):
        yield child


def _schema_children_with_paths(
    node: dict[str, Any],
) -> Iterator[tuple[tuple[str, ...], Any]]:
    for keyword in _SCHEMA_MAP_KEYWORDS:
        children = node.get(keyword)
        if isinstance(children, dict):
            for name, child in cast(dict[str, Any], children).items():
                yield (keyword, name), child

    for keyword in _SCHEMA_VALUE_KEYWORDS:
        if keyword in node:
            yield (keyword,), node[keyword]


def _unsupported(path: tuple[str, ...], detail: str) -> None:
    location = "/".join(path) or "<root>"
    raise ValueError(
        f"LLM schema is not compatible with strict structured output at "
        f"{location}: {detail}"
    )
