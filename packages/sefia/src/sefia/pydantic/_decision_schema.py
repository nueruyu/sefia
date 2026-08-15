from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from pydantic import TypeAdapter, WithJsonSchema

from ..llm.schema import LLMSchema, SchemaPath
from ._tool_arguments import ToolArgumentContract, ToolSchemaKind

_TOOL_ARGUMENT_MARKER = "x-sefia-tool-arguments"


@dataclass(frozen=True)
class _ComposedSchema:
    schema: dict[str, Any]
    definitions: dict[str, dict[str, Any]]


def tool_argument_placeholder(tool_name: str) -> WithJsonSchema:
    return WithJsonSchema({_TOOL_ARGUMENT_MARKER: tool_name})


def build_decision_schema(
    model: Any, tools: dict[str, ToolArgumentContract]
) -> LLMSchema:
    schema = TypeAdapter(model).json_schema()
    definitions = schema.setdefault("$defs", {})
    if not isinstance(definitions, dict):
        raise ValueError("LLM schema $defs must be an object")
    root_definitions = cast(dict[str, Any], definitions)
    raw_nodes: set[int] = set()
    raw_definition_names: set[str] = set()

    for node in list(_walk(schema)):
        tool_name = node.pop(_TOOL_ARGUMENT_MARKER, None)
        if not isinstance(tool_name, str):
            continue
        if tool_name not in tools:
            raise ValueError(f"Unknown tool argument marker: {tool_name!r}")
        contract = tools[tool_name]
        composed = _compose_definitions(
            root_definitions,
            tool_name,
            deepcopy(contract.schema),
            protected_names=raw_definition_names,
            preserve=contract.kind is ToolSchemaKind.RAW,
        )
        node.clear()
        node.update(composed.schema)
        if contract.kind is ToolSchemaKind.RAW:
            raw_nodes.add(id(node))
            raw_nodes.update(
                id(definition) for definition in composed.definitions.values()
            )
            raw_definition_names.update(composed.definitions)

    raw_paths = frozenset(
        path for path, node in _walk_with_paths(schema) if id(node) in raw_nodes
    )
    schema["description"] = "The model for the LLM's decision on the next action."
    return LLMSchema(schema=schema, raw_schema_paths=raw_paths)


def _compose_definitions(
    root: dict[str, Any],
    namespace: str,
    schema: dict[str, Any],
    *,
    protected_names: set[str],
    preserve: bool,
) -> _ComposedSchema:
    local: dict[str, Any] = {}
    for keyword in ("$defs", "definitions"):
        definitions = schema.pop(keyword, None)
        if isinstance(definitions, dict):
            local.update(cast(dict[str, Any], definitions))

    names = {
        name: _target_name(
            root,
            namespace,
            name,
            definition,
            protected_names=protected_names,
            preserve=preserve,
        )
        for name, definition in local.items()
    }
    _rewrite_references(schema, names)
    composed: dict[str, dict[str, Any]] = {}
    for name, definition in local.items():
        target = names[name]
        if target not in root:
            rewritten = deepcopy(definition)
            _rewrite_references(rewritten, names)
            root[target] = rewritten
            composed[target] = rewritten
    return _ComposedSchema(schema, composed)


def _target_name(
    root: dict[str, Any],
    namespace: str,
    name: str,
    definition: Any,
    *,
    protected_names: set[str],
    preserve: bool,
) -> str:
    if name not in root:
        return name
    if not preserve and name not in protected_names and root[name] == definition:
        return name
    base = f"{namespace}__{name}"
    candidate = base
    suffix = 2
    while candidate in root:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _rewrite_references(node: Any, names: dict[str, str]) -> None:
    for schema_node in _walk(node):
        reference = schema_node.get("$ref")
        if not isinstance(reference, str):
            continue
        for prefix in ("#/$defs/", "#/definitions/"):
            if reference.startswith(prefix):
                name = reference.removeprefix(prefix)
                if name in names:
                    schema_node["$ref"] = f"#/$defs/{names[name]}"
                break


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    for _, child in _walk_with_paths(node):
        yield child


def _walk_with_paths(
    node: Any, path: SchemaPath = ()
) -> Iterator[tuple[SchemaPath, dict[str, Any]]]:
    if isinstance(node, list):
        for index, item in enumerate(cast(list[Any], node)):
            yield from _walk_with_paths(item, (*path, index))
        return
    if not isinstance(node, dict):
        return
    schema = cast(dict[str, Any], node)
    yield path, schema
    for key, child in schema.items():
        yield from _walk_with_paths(child, (*path, key))
