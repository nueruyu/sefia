from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, WithJsonSchema

from ..llm.schema import (
    JsonObject,
    JsonSchemaDocument,
    JsonValue,
    LLMSchema,
    SchemaKeyword,
    SchemaNode,
    SchemaPath,
    require_json_object,
)
from ._tool_arguments import ToolArgumentContract, ToolSchemaKind

_TOOL_ARGUMENT_MARKER = "x-sefia-tool-arguments"
K = SchemaKeyword


@dataclass(frozen=True)
class _ComposedSchema:
    schema: JsonObject
    definitions: JsonObject


def tool_argument_placeholder(tool_name: str) -> WithJsonSchema:
    return WithJsonSchema({_TOOL_ARGUMENT_MARKER: tool_name})


def build_decision_schema(
    model: Any, tools: dict[str, ToolArgumentContract]
) -> LLMSchema:
    schema = require_json_object(TypeAdapter(model).json_schema())
    root = SchemaNode(schema)
    root_definitions = root.ensure_definitions()
    raw_paths: set[SchemaPath] = set()
    raw_definition_names: set[str] = set()

    for path, node in list(_walk_with_paths(schema)):
        tool_name = node.pop(_TOOL_ARGUMENT_MARKER, None)
        if not isinstance(tool_name, str):
            continue
        if tool_name not in tools:
            raise ValueError(f"Unknown tool argument marker: {tool_name!r}")
        contract = tools[tool_name]
        composed = _compose_definitions(
            root_definitions,
            tool_name,
            contract.schema.mutable_copy(),
            protected_names=raw_definition_names,
            preserve=contract.kind is ToolSchemaKind.RAW,
        )
        node.clear()
        node.update(composed.schema)
        if contract.kind is ToolSchemaKind.RAW:
            raw_paths.add(path)
            raw_paths.update((K.DEFINITIONS, name) for name in composed.definitions)
            raw_definition_names.update(composed.definitions)

    root.set_description("The model for the LLM's decision on the next action.")
    return LLMSchema(
        document=JsonSchemaDocument.from_mapping(schema),
        raw_schema_paths=frozenset(raw_paths),
    )


def _compose_definitions(
    root: JsonObject,
    namespace: str,
    schema: JsonObject,
    *,
    protected_names: set[str],
    preserve: bool,
) -> _ComposedSchema:
    local = SchemaNode(schema).take_definitions()

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
    composed: JsonObject = {}
    for name, definition in local.items():
        target = names[name]
        if target not in root:
            rewritten = deepcopy(definition)
            _rewrite_references(rewritten, names)
            root[target] = rewritten
            composed[target] = rewritten
    return _ComposedSchema(schema, composed)


def _target_name(
    root: JsonObject,
    namespace: str,
    name: str,
    definition: JsonValue,
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


def _rewrite_references(node: JsonValue, names: dict[str, str]) -> None:
    for schema_node in _walk(node):
        target = SchemaNode(schema_node)
        reference = target.local_reference
        if reference is None:
            continue
        if reference.definition in names:
            target.set_local_reference(
                reference.with_definition(names[reference.definition])
            )


def _walk(node: JsonValue) -> Iterator[JsonObject]:
    for _, child in _walk_with_paths(node):
        yield child


def _walk_with_paths(
    node: JsonValue, path: SchemaPath = ()
) -> Iterator[tuple[SchemaPath, JsonObject]]:
    if isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk_with_paths(item, (*path, index))
        return
    if not isinstance(node, dict):
        return
    yield path, node
    for key, child in node.items():
        yield from _walk_with_paths(child, (*path, key))
