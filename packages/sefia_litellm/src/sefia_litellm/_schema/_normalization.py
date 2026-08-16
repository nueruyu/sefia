from dataclasses import dataclass

from typing_extensions import final

from sefia.llm.schema import JsonObject, SchemaNode, SchemaPath

from ._traversal import walk

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
    def __init__(self, preserved: frozenset[SchemaPath]):
        self._preserved = preserved

    def normalize(self, schema: JsonObject) -> None:
        for _, node in walk(schema, skip=self._preserved):
            if node.type == "object":
                node.value.setdefault("additionalProperties", False)
                properties = node.object_map("properties")
                if properties is not None:
                    node.value["required"] = list(properties)
            one_of = node.value.pop("oneOf", None)
            if one_of is not None:
                node.value["anyOf"] = one_of


@final
@dataclass(frozen=True)
class MappingEncoding:
    path: SchemaPath


@final
@dataclass(frozen=True)
class SchemaEncodingPlan:
    mappings: tuple[MappingEncoding, ...] = ()

    @property
    def mapping_paths(self) -> frozenset[SchemaPath]:
        return frozenset(encoding.path for encoding in self.mappings)


@final
class MappingLowerer:
    def __init__(self, preserved: frozenset[SchemaPath]):
        self._preserved = preserved

    def lower(self, schema: JsonObject) -> SchemaEncodingPlan:
        mappings: list[MappingEncoding] = []
        for path, node in list(walk(schema, skip=self._preserved)):
            additional = node.additional_properties()
            if node.type != "object" or not isinstance(additional, SchemaNode):
                continue
            property_names = node.child("propertyNames")
            key_schema: JsonObject = (
                property_names.value
                if property_names is not None
                else {"type": "string"}
            )
            lowered: JsonObject = {
                key: node.value[key]
                for key in ("title", "description")
                if key in node.value
            }
            if "minProperties" in node.value:
                lowered["minItems"] = node.value["minProperties"]
            if "maxProperties" in node.value:
                lowered["maxItems"] = node.value["maxProperties"]
            lowered.update(
                type="array",
                items={
                    "type": "object",
                    "properties": {
                        "key": key_schema,
                        "value": additional.value,
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            )
            node.value.clear()
            node.value.update(lowered)
            mappings.append(MappingEncoding(path))
        return SchemaEncodingPlan(tuple(mappings))


@final
class CompatibilityValidator:
    def validate(self, schema: JsonObject) -> None:
        for path, node in walk(schema):
            if "oneOf" in node.value:
                self._unsupported(path, "oneOf is not supported; use a disjoint anyOf")
            for keyword in _UNSUPPORTED_COMPOSITION:
                if keyword in node.value:
                    self._unsupported(path, f"{keyword} is not supported")
            if node.type == "object":
                self._validate_object(path, node)

    def _validate_object(self, path: SchemaPath, node: SchemaNode) -> None:
        if node.additional_properties() is not False:
            self._unsupported(
                path, "object schemas must set additionalProperties to false"
            )
        properties = node.object_map("properties")
        if properties is None:
            return
        required = node.value.get("required")
        required_names: set[str] = (
            {item for item in required if isinstance(item, str)}
            if isinstance(required, list)
            else set()
        )
        missing = sorted(set(properties) - required_names)
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
