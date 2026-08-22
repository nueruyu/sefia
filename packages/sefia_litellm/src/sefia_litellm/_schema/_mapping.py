from dataclasses import dataclass
from typing_extensions import final

from sefia.llm.json_schema import (
    JsonObject,
    JsonSchemaDocument,
    SchemaKeyword,
    SchemaNode,
    SchemaPath,
)
from sefia.llm.llm_output import LLMOutput

from ._mapping_restoration import restore_mappings

K = SchemaKeyword

_PRESERVED_ANNOTATIONS = (K.TITLE, K.DESCRIPTION)
_PROPERTY_TO_ITEM_CONSTRAINT = {
    K.MIN_PROPERTIES: K.MIN_ITEMS,
    K.MAX_PROPERTIES: K.MAX_ITEMS,
}


@final
@dataclass(frozen=True)
class DictionarySchema:
    key_schema: JsonObject
    value_schema: JsonObject
    annotations: JsonObject
    entry_array_constraints: JsonObject

    @classmethod
    def from_node(cls, node: SchemaNode) -> "DictionarySchema":
        values = node.additional_properties()
        if node.type != "object" or not isinstance(values, SchemaNode):
            raise ValueError("node is not a dictionary schema")
        if node.properties() or node.required():
            raise ValueError(
                "objects combining fixed properties with dictionary values "
                "cannot be lowered safely"
            )
        keys = node.property_names()
        return cls(
            key_schema=keys.value if keys is not None else {K.TYPE: "string"},
            value_schema=values.value,
            annotations={
                keyword: node.value[keyword]
                for keyword in _PRESERVED_ANNOTATIONS
                if keyword in node.value
            },
            entry_array_constraints={
                target: node.value[source]
                for source, target in _PROPERTY_TO_ITEM_CONSTRAINT.items()
                if source in node.value
            },
        )

    def to_entry_array_schema(self) -> JsonObject:
        entry = SchemaNode.object_schema(
            {"key": self.key_schema, "value": self.value_schema}
        )
        return {
            **self.annotations,
            **self.entry_array_constraints,
            K.TYPE: "array",
            K.ITEMS: entry.value,
        }


@final
@dataclass(frozen=True)
class LoweredMappingSchema:
    wire_schema: JsonObject
    restoration_paths: frozenset[SchemaPath]

    def decode(self, output: LLMOutput) -> LLMOutput:
        return restore_mappings(
            output,
            schema=self.wire_schema,
            restoration_paths=self.restoration_paths,
        )


def lower_mapping_schemas(schema: JsonObject) -> LoweredMappingSchema:
    restoration_paths: set[SchemaPath] = set()
    while cursor := _find_dictionary_schema(schema):
        path, node, dictionary = cursor
        replacement = dictionary.to_entry_array_schema()
        node.value.clear()
        node.value.update(replacement)
        restoration_paths.add(path)
    wire_schema = JsonSchemaDocument(schema).mutable_copy()
    return LoweredMappingSchema(wire_schema, frozenset(restoration_paths))


def _find_dictionary_schema(
    schema: JsonObject,
) -> tuple[SchemaPath, SchemaNode, DictionarySchema] | None:
    for cursor in SchemaNode(schema).walk():
        if _is_dictionary_schema(cursor.node):
            return (
                cursor.path,
                cursor.node,
                DictionarySchema.from_node(cursor.node),
            )
    return None


def _is_dictionary_schema(node: SchemaNode) -> bool:
    return node.type == "object" and isinstance(
        node.additional_properties(), SchemaNode
    )
