from dataclasses import dataclass

from typing_extensions import final

from sefia.llm.json_schema import JsonObject, JsonSchemaDocument
from sefia.llm.llm_output import LLMOutput
from sefia.llm.step_decision import StepTool, TypedToolArguments

from ._mapping import UniformDictionaryFormat
from ._policy import (
    GENERATED_SCHEMA_POLICY,
    USER_DEFINED_SCHEMA_POLICY,
    SchemaPolicy,
    prepare_schema,
)


@final
@dataclass(frozen=True)
class StructuredValueFormat:
    schema: JsonObject
    dictionary_format: UniformDictionaryFormat | None

    @classmethod
    def from_generated_schema(
        cls, document: JsonSchemaDocument
    ) -> "StructuredValueFormat":
        return cls._from_schema(document, GENERATED_SCHEMA_POLICY)

    @classmethod
    def from_user_schema(cls, document: JsonSchemaDocument) -> "StructuredValueFormat":
        return cls._from_schema(document, USER_DEFINED_SCHEMA_POLICY)

    @classmethod
    def from_tool(cls, tool: StepTool) -> "StructuredValueFormat":
        if isinstance(tool.arguments, TypedToolArguments):
            return cls.from_generated_schema(tool.arguments.json_schema)
        return cls.from_user_schema(tool.arguments.json_schema)

    @classmethod
    def _from_schema(
        cls, document: JsonSchemaDocument, policy: SchemaPolicy
    ) -> "StructuredValueFormat":
        prepared = prepare_schema(document.mutable_copy(), policy)
        return cls(prepared.wire_schema, prepared.dictionary_format)

    def decode(self, output: LLMOutput) -> LLMOutput:
        if self.dictionary_format is None:
            return output
        return self.dictionary_format.decode(output)
