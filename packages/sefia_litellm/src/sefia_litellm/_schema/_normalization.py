from dataclasses import dataclass

from typing_extensions import final

from ._traversal import walk
from sefia.llm.schema import JsonObject, SchemaKeyword, SchemaNode, SchemaPath

K = SchemaKeyword

_UNSUPPORTED_COMPOSITION = (
    K.ALL_OF,
    K.NOT,
    K.DEPENDENT_REQUIRED,
    K.DEPENDENT_SCHEMAS,
    K.IF,
    K.THEN,
    K.ELSE,
)


@final
class SchemaNormalizer:
    def __init__(self, preserved: frozenset[SchemaPath]):
        self._preserved = preserved

    def normalize(self, schema: JsonObject) -> None:
        for _, node in walk(schema, skip=self._preserved):
            if node.type == "object":
                node.close_object()
            node.normalize_one_of()


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
            property_names = node.property_names()
            key_schema: JsonObject = (
                property_names.value
                if property_names is not None
                else {K.TYPE: "string"}
            )
            node.replace_with_mapping_entries(key_schema, additional.value)
            mappings.append(MappingEncoding(path))
        return SchemaEncodingPlan(tuple(mappings))


@final
class CompatibilityValidator:
    def validate(self, schema: JsonObject) -> None:
        for path, node in walk(schema):
            if K.ONE_OF in node.value:
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
        properties = node.properties()
        required_names = set(node.required() or ())
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
