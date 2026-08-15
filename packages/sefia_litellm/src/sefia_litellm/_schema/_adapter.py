from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, cast

from typing_extensions import final, override

from sefia.llm.schema import LLMSchema, PreparedLLMSchema, SchemaPath
from ._traversal import matches, resolve, walk, walk_with_paths

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
class LiteLLMPreparedSchema(PreparedLLMSchema):
    _schema: dict[str, Any]
    _decoder: _Decoder

    @property
    @override
    def schema(self) -> dict[str, Any]:
        return deepcopy(self._schema)

    @override
    def decode(self, data: Any) -> Any:
        decoded = self._decoder.decode(data)
        if not isinstance(decoded, dict):
            return decoded
        decoded_map = cast(dict[str, Any], decoded)
        if set(decoded_map) == {"payload"}:
            return decoded_map["payload"]
        return decoded_map

    @override
    def normalize_stream_path(self, path: SchemaPath) -> SchemaPath | None:
        if path and path[0] == "payload":
            return path[1:]
        return path


@final
class LiteLLMSchemaAdapter:
    """Build a provider schema and the inverse response decoder as one contract."""

    def build(
        self,
        logical: LLMSchema,
    ) -> LiteLLMPreparedSchema:
        schema, preserved = _EnvelopeComposer().compose(logical)
        _SchemaNormalizer(preserved).normalize(schema)
        mapping_ids = _MappingLowerer(preserved).lower(schema)
        _CompatibilityValidator().validate(schema)
        schema["description"] = "The model for the LLM's decision on the next action."
        return LiteLLMPreparedSchema(
            schema, _DecoderFactory(schema, mapping_ids).build(schema)
        )


@final
class _EnvelopeComposer:
    """Wrap a logical decision schema for strict structured output."""

    def compose(self, logical: LLMSchema) -> tuple[dict[str, Any], set[int]]:
        payload = deepcopy(logical.schema)
        preserved = {
            id(node)
            for path, node in walk_with_paths(payload)
            if path in logical.raw_schema_paths
        }
        definitions = payload.pop("$defs", None)
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"payload": payload},
            "required": ["payload"],
            "additionalProperties": False,
        }
        if isinstance(definitions, dict):
            schema["$defs"] = definitions
        return schema, preserved


@final
class _SchemaNormalizer:
    """Apply non-semantic provider normalization to generated schemas."""

    def __init__(self, preserved: set[int]):
        self._preserved = preserved

    def normalize(self, schema: dict[str, Any]) -> None:
        for node in walk(schema, skip=self._preserved):
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
        for node in walk(schema, skip=self._preserved):
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
        for path, node in walk_with_paths(schema):
            if "oneOf" in node:
                self._unsupported(path, "oneOf is not supported; use a disjoint anyOf")
            for keyword in _UNSUPPORTED_COMPOSITION:
                if keyword in node:
                    self._unsupported(path, f"{keyword} is not supported")
            if node.get("type") == "object":
                self._validate_object(path, node)

    def _validate_object(self, path: SchemaPath, node: dict[str, Any]) -> None:
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
    def _unsupported(path: SchemaPath, detail: str) -> None:
        location = "/".join(map(str, path)) or "<root>"
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
            if matches(data, schema, self._root):
                return decoder.decode(data)
        return data


@final
class _DecoderFactory:
    def __init__(self, root: dict[str, Any], mapping_ids: set[int]):
        self._root = root
        self._mapping_ids = mapping_ids
        self._cache: dict[int, _DeferredDecoder] = {}

    def build(self, schema: dict[str, Any]) -> _Decoder:
        schema = resolve(schema, self._root)
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
