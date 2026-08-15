from typing import Any, cast

from typing_extensions import final

from sefia.llm.schema import SchemaPath

from ._traversal import walk, walk_with_paths

_UNSUPPORTED_COMPOSITION = (
    "allOf",
    "not",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
)


@final
class SchemaNormalizer:
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
class MappingLowerer:
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
            lowered = {
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
class CompatibilityValidator:
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
            "LLM schema is not compatible with strict structured output at "
            f"{location}: {detail}"
        )
