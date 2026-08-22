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
class MappingSchema:
    key_schema: JsonObject
    value_schema: JsonObject
    annotations: JsonObject
    entry_list_constraints: JsonObject

    @classmethod
    def from_node(cls, node: SchemaNode) -> "MappingSchema | None":
        values = node.additional_properties()
        if node.type != "object" or not isinstance(values, SchemaNode):
            return None
        keys = node.property_names()
        return cls(
            key_schema=keys.value if keys is not None else {K.TYPE: "string"},
            value_schema=values.value,
            annotations={
                keyword: node.value[keyword]
                for keyword in _PRESERVED_ANNOTATIONS
                if keyword in node.value
            },
            entry_list_constraints={
                target: node.value[source]
                for source, target in _PROPERTY_TO_ITEM_CONSTRAINT.items()
                if source in node.value
            },
        )

    def to_entry_list(self) -> "MappingEntryListSchema":
        return MappingEntryListSchema(
            key_schema=self.key_schema,
            value_schema=self.value_schema,
            annotations=self.annotations,
            constraints=self.entry_list_constraints,
        )


@final
@dataclass(frozen=True)
class MappingEntryListSchema:
    key_schema: JsonObject
    value_schema: JsonObject
    annotations: JsonObject
    constraints: JsonObject

    def to_json_schema(self) -> JsonObject:
        entry = SchemaNode.object_schema(
            {"key": self.key_schema, "value": self.value_schema}
        )
        return {
            **self.annotations,
            **self.constraints,
            K.TYPE: "array",
            K.ITEMS: entry.value,
        }


@final
@dataclass(frozen=True)
class MappingTransform:
    schema: JsonObject
    mapping_paths: frozenset[SchemaPath]

    @classmethod
    def lower(cls, schema: JsonObject) -> "MappingTransform":
        mapping_paths: set[SchemaPath] = set()
        for cursor in list(SchemaNode(schema).walk()):
            path, node = cursor.path, cursor.node
            mapping = MappingSchema.from_node(node)
            if mapping is None:
                continue
            replacement = mapping.to_entry_list().to_json_schema()
            node.value.clear()
            node.value.update(replacement)
            mapping_paths.add(path)
        restoration_schema = JsonSchemaDocument(schema).mutable_copy()
        return cls(restoration_schema, frozenset(mapping_paths))

    def restore(self, output: LLMOutput) -> LLMOutput:
        return restore_mappings(
            output,
            schema=self.schema,
            mapping_paths=self.mapping_paths,
        )
