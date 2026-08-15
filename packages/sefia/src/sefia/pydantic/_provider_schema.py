from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, cast

from pydantic import ConfigDict, TypeAdapter, create_model
from typing_extensions import final

from ._tool_arguments import ToolArgumentContract, ToolSchemaKind

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
_UNSUPPORTED_COMPOSITION = (
    "allOf",
    "not",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
)


class _Decoder(Protocol):
    def decode(self, data: Any) -> Any: ...


@final
@dataclass
class ProviderSchema:
    schema: dict[str, Any]
    _decoder: _Decoder

    def decode(self, data: Any) -> Any:
        return self._decoder.decode(data)


@final
class ProviderSchemaBuilder:
    """Build a provider schema and the inverse response decoder as one contract."""

    def build(
        self,
        model: Any,
        *,
        name: str,
        tools: dict[str, ToolArgumentContract],
    ) -> ProviderSchema:
        schema, preserved = _EnvelopeComposer(name, tools).compose(model)
        _SchemaNormalizer(preserved).normalize(schema)
        mapping_ids = _MappingLowerer(preserved).lower(schema)
        _CompatibilityValidator().validate(schema)
        schema["description"] = "The model for the LLM's decision on the next action."
        return ProviderSchema(
            schema, _DecoderFactory(schema, mapping_ids).build(schema)
        )


@final
class _EnvelopeComposer:
    """Compose Pydantic's envelope with the original per-tool schemas."""

    def __init__(self, name: str, tools: dict[str, ToolArgumentContract]):
        self._name = name
        self._tools = tools

    def compose(self, model: Any) -> tuple[dict[str, Any], set[int]]:
        envelope = create_model(
            f"{self._name}Envelope",
            __config__=ConfigDict(extra="forbid"),
            payload=(model, ...),
        )
        schema = TypeAdapter(envelope).json_schema()
        definitions = schema.setdefault("$defs", {})
        if not isinstance(definitions, dict):
            raise ValueError("LLM schema $defs must be an object")
        root_defs = cast(dict[str, Any], definitions)
        preserved: set[int] = set()

        for node in list(_walk(schema)):
            match = self._tool_call(node)
            if match is None:
                continue
            tool_name, properties = match
            contract = self._tools[tool_name]
            arguments = deepcopy(contract.schema)
            if contract.kind is ToolSchemaKind.RAW:
                preserved.add(id(arguments))
            else:
                arguments = _DefinitionComposer(root_defs, tool_name).compose(arguments)
            properties["arguments"] = arguments

        return schema, preserved

    def _tool_call(self, node: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        properties = node.get("properties")
        if not isinstance(properties, dict):
            return None
        property_map = cast(dict[str, Any], properties)
        name_schema = property_map.get("name")
        if not isinstance(name_schema, dict):
            return None
        tool_name = cast(dict[str, Any], name_schema).get("const")
        if not isinstance(tool_name, str) or tool_name not in self._tools:
            return None
        if not isinstance(property_map.get("arguments"), dict):
            return None
        return tool_name, property_map


@final
class _DefinitionComposer:
    """Hoist local definitions and rewrite colliding references."""

    def __init__(self, root_definitions: dict[str, Any], namespace: str):
        self._root = root_definitions
        self._namespace = namespace

    def compose(self, schema: dict[str, Any]) -> dict[str, Any]:
        local: dict[str, Any] = {}
        for keyword in ("$defs", "definitions"):
            definitions = schema.pop(keyword, None)
            if isinstance(definitions, dict):
                local.update(cast(dict[str, Any], definitions))

        names = {
            name: self._target_name(name, definition)
            for name, definition in local.items()
        }
        self._rewrite_references(schema, names)
        for name, definition in local.items():
            target = names[name]
            if target in self._root:
                continue
            rewritten = deepcopy(definition)
            self._rewrite_references(rewritten, names)
            self._root[target] = rewritten
        return schema

    def _target_name(self, name: str, definition: Any) -> str:
        if name not in self._root or self._root[name] == definition:
            return name
        base = f"{self._namespace}__{name}"
        candidate = base
        suffix = 2
        while candidate in self._root:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    @staticmethod
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


@final
class _SchemaNormalizer:
    """Apply non-semantic provider normalization to generated schemas."""

    def __init__(self, preserved: set[int]):
        self._preserved = preserved

    def normalize(self, schema: dict[str, Any]) -> None:
        for node in _walk(schema, skip=self._preserved):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
                properties = node.get("properties")
                if isinstance(properties, dict):
                    node["required"] = list(cast(dict[str, Any], properties))
            one_of = node.pop("oneOf", None)
            if one_of is not None:
                node["anyOf"] = one_of


@final
class _MappingLowerer:
    """Encode typed dynamic objects as strict-provider entry arrays."""

    def __init__(self, preserved: set[int]):
        self._preserved = preserved

    def lower(self, schema: dict[str, Any]) -> set[int]:
        mapping_ids: set[int] = set()
        for node in _walk(schema, skip=self._preserved):
            additional = node.get("additionalProperties")
            if node.get("type") != "object" or not isinstance(additional, dict):
                continue
            value_schema = cast(dict[str, Any], additional)
            property_names = node.get("propertyNames")
            key_schema = (
                cast(dict[str, Any], property_names)
                if isinstance(property_names, dict)
                else {"type": "string"}
            )
            lowered: dict[str, Any] = {
                key: node[key] for key in ("title", "description") if key in node
            }
            if "minProperties" in node:
                lowered["minItems"] = node["minProperties"]
            if "maxProperties" in node:
                lowered["maxItems"] = node["maxProperties"]
            lowered.update(
                type="array",
                items={
                    "type": "object",
                    "properties": {"key": key_schema, "value": value_schema},
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            )
            node.clear()
            node.update(lowered)
            mapping_ids.add(id(node))
        return mapping_ids


@final
class _CompatibilityValidator:
    def validate(self, schema: dict[str, Any]) -> None:
        for path, node in _walk_with_paths(schema):
            if "oneOf" in node:
                self._unsupported(path, "oneOf is not supported; use a disjoint anyOf")
            for keyword in _UNSUPPORTED_COMPOSITION:
                if keyword in node:
                    self._unsupported(path, f"{keyword} is not supported")
            if node.get("type") == "object":
                self._validate_object(path, node)

    def _validate_object(self, path: tuple[str, ...], node: dict[str, Any]) -> None:
        if node.get("additionalProperties") is not False:
            self._unsupported(
                path, "object schemas must set additionalProperties to false"
            )
        properties = node.get("properties")
        if not isinstance(properties, dict):
            return
        property_names = set(cast(dict[str, Any], properties))
        required = node.get("required")
        required_names: set[str] = (
            set(cast(list[str], required)) if isinstance(required, list) else set()
        )
        missing = sorted(property_names - required_names)
        if missing:
            self._unsupported(
                path, f"all object properties must be required; missing {missing}"
            )

    @staticmethod
    def _unsupported(path: tuple[str, ...], detail: str) -> None:
        location = "/".join(path) or "<root>"
        raise ValueError(
            f"LLM schema is not compatible with strict structured output at "
            f"{location}: {detail}"
        )


@final
class _IdentityDecoder:
    def decode(self, data: Any) -> Any:
        return data


@final
class _DeferredDecoder:
    def __init__(self) -> None:
        self.target: _Decoder | None = None

    def decode(self, data: Any) -> Any:
        return data if self.target is None else self.target.decode(data)


@final
class _ObjectDecoder:
    def __init__(self, properties: dict[str, _Decoder]):
        self._properties = properties

    def decode(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return {
            key: self._properties[key].decode(value)
            if key in self._properties
            else value
            for key, value in cast(dict[str, Any], data).items()
        }


@final
class _ArrayDecoder:
    def __init__(self, item: _Decoder):
        self._item = item

    def decode(self, data: Any) -> Any:
        if not isinstance(data, list):
            return data
        return [self._item.decode(item) for item in cast(list[Any], data)]


@final
class _MappingDecoder:
    def __init__(self, key: _Decoder, value: _Decoder):
        self._key = key
        self._value = value

    def decode(self, data: Any) -> Any:
        if not isinstance(data, list):
            return data
        result: dict[Any, Any] = {}
        for entry in cast(list[Any], data):
            if not isinstance(entry, dict):
                raise ValueError("mapping entries must be objects")
            entry_map = cast(dict[str, Any], entry)
            if set(entry_map) != {"key", "value"}:
                raise ValueError("mapping entries must contain only key and value")
            key = self._key.decode(entry_map["key"])
            try:
                if key in result:
                    raise ValueError(f"duplicate mapping key: {key!r}")
                result[key] = self._value.decode(entry_map["value"])
            except TypeError as error:
                raise ValueError("mapping keys must be hashable") from error
        return result


@final
class _UnionDecoder:
    def __init__(
        self, choices: list[tuple[dict[str, Any], _Decoder]], root: dict[str, Any]
    ):
        self._choices = choices
        self._root = root

    def decode(self, data: Any) -> Any:
        for schema, decoder in self._choices:
            if _matches(data, schema, self._root):
                return decoder.decode(data)
        return data


@final
class _DecoderFactory:
    def __init__(self, root: dict[str, Any], mapping_ids: set[int]):
        self._root = root
        self._mapping_ids = mapping_ids
        self._cache: dict[int, _DeferredDecoder] = {}

    def build(self, schema: dict[str, Any]) -> _Decoder:
        schema = _resolve(schema, self._root)
        cached = self._cache.get(id(schema))
        if cached is not None:
            return cached
        deferred = _DeferredDecoder()
        self._cache[id(schema)] = deferred
        deferred.target = self._build_node(schema)
        return deferred

    def _build_node(self, schema: dict[str, Any]) -> _Decoder:
        if id(schema) in self._mapping_ids:
            properties = cast(
                dict[str, Any], cast(dict[str, Any], schema["items"])["properties"]
            )
            return _MappingDecoder(
                self.build(cast(dict[str, Any], properties["key"])),
                self.build(cast(dict[str, Any], properties["value"])),
            )
        any_of = schema.get("anyOf")
        if isinstance(any_of, list):
            choices = [
                (candidate, self.build(candidate))
                for item in cast(list[Any], any_of)
                if isinstance(item, dict)
                for candidate in [cast(dict[str, Any], item)]
            ]
            return _UnionDecoder(choices, self._root)
        if schema.get("type") == "object":
            properties = schema.get("properties")
            if isinstance(properties, dict):
                return _ObjectDecoder(
                    {
                        name: self.build(cast(dict[str, Any], child))
                        for name, child in cast(dict[str, Any], properties).items()
                        if isinstance(child, dict)
                    }
                )
        if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
            return _ArrayDecoder(self.build(cast(dict[str, Any], schema["items"])))
        return _IdentityDecoder()


def _walk(node: Any, *, skip: set[int] | None = None) -> Iterator[dict[str, Any]]:
    if isinstance(node, list):
        for item in cast(list[Any], node):
            yield from _walk(item, skip=skip)
        return
    if not isinstance(node, dict):
        return
    schema = cast(dict[str, Any], node)
    if skip is not None and id(schema) in skip:
        return
    yield schema
    for _, child in _children(schema):
        yield from _walk(child, skip=skip)


def _walk_with_paths(
    node: Any, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    if isinstance(node, list):
        for index, item in enumerate(cast(list[Any], node)):
            yield from _walk_with_paths(item, (*path, str(index)))
        return
    if not isinstance(node, dict):
        return
    schema = cast(dict[str, Any], node)
    yield path, schema
    for child_path, child in _children(schema):
        yield from _walk_with_paths(child, (*path, *child_path))


def _children(node: dict[str, Any]) -> Iterator[tuple[tuple[str, ...], Any]]:
    for keyword in _MAP_CHILDREN:
        children = node.get(keyword)
        if isinstance(children, dict):
            for name, child in list(cast(dict[str, Any], children).items()):
                yield (keyword, name), child
    for keyword in _VALUE_CHILDREN:
        if keyword in node:
            yield (keyword,), node[keyword]


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
        return schema
    definitions = root.get("$defs")
    if not isinstance(definitions, dict):
        return schema
    name = reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
    resolved = cast(dict[str, Any], definitions).get(name)
    return cast(dict[str, Any], resolved) if isinstance(resolved, dict) else schema


def _matches(data: Any, schema: dict[str, Any], root: dict[str, Any]) -> bool:
    schema = _resolve(schema, root)
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
