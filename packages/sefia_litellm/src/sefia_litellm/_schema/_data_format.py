from dataclasses import dataclass

from typing_extensions import final

from sefia.llm.json_schema import JsonObject, JsonSchemaDocument
from sefia.llm.structured_data import StructuredData
from sefia.llm.step_decision import StepTool, ToolSchemaSource

from ._uniform_dictionary import UniformDictionaryFormat
from ._policy import (
    GENERATED_SCHEMA_POLICY,
    USER_DEFINED_SCHEMA_POLICY,
    SchemaPolicy,
    prepare_schema,
)


@final
@dataclass(frozen=True)
class StructuredDataFormat:
    schema: JsonObject
    dictionary_format: UniformDictionaryFormat | None

    @property
    def transforms_data(self) -> bool:
        return self.dictionary_format is not None

    @classmethod
    def from_generated_schema(
        cls, document: JsonSchemaDocument
    ) -> "StructuredDataFormat":
        return cls._from_schema(document, GENERATED_SCHEMA_POLICY)

    @classmethod
    def from_user_schema(cls, document: JsonSchemaDocument) -> "StructuredDataFormat":
        return cls._from_schema(document, USER_DEFINED_SCHEMA_POLICY)

    @classmethod
    def from_tool(cls, tool: StepTool) -> "StructuredDataFormat":
        if tool.schema_source is ToolSchemaSource.GENERATED:
            return cls.from_generated_schema(tool.arguments)
        return cls.from_user_schema(tool.arguments)

    @classmethod
    def _from_schema(
        cls, document: JsonSchemaDocument, policy: SchemaPolicy
    ) -> "StructuredDataFormat":
        prepared = prepare_schema(document.mutable_copy(), policy)
        return cls(prepared.wire_schema, prepared.dictionary_format)

    def decode(self, data: StructuredData) -> StructuredData:
        if self.dictionary_format is None:
            return data
        return self.dictionary_format.decode(data)

    def encode(self, data: StructuredData) -> StructuredData:
        if self.dictionary_format is None:
            return data
        return self.dictionary_format.encode(data)
