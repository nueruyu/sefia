from dataclasses import dataclass

from typing_extensions import final

from sefia.llm.json_schema import JsonObject, JsonSchemaDocument
from sefia.llm.llm_output import LLMOutput

from ._uniform_dictionary import UniformDictionaryFormat
from ._policy import (
    GENERATED_SCHEMA_POLICY,
    USER_DEFINED_SCHEMA_POLICY,
    SchemaPolicy,
    apply_schema_policy,
)


@final
@dataclass(frozen=True)
class StructuredValueFormat:
    schema: JsonObject
    dictionary_format: UniformDictionaryFormat | None

    @property
    def translates_values(self) -> bool:
        return self.dictionary_format is not None

    @classmethod
    def from_generated_schema(
        cls, document: JsonSchemaDocument
    ) -> "StructuredValueFormat":
        return cls._from_schema(document, GENERATED_SCHEMA_POLICY)

    @classmethod
    def from_user_schema(cls, document: JsonSchemaDocument) -> "StructuredValueFormat":
        return cls._from_schema(document, USER_DEFINED_SCHEMA_POLICY)

    @classmethod
    def _from_schema(
        cls, document: JsonSchemaDocument, policy: SchemaPolicy
    ) -> "StructuredValueFormat":
        schema = document.mutable_copy()
        dictionary_format = apply_schema_policy(schema, policy)
        return cls(schema, dictionary_format)

    def decode(self, output: LLMOutput) -> LLMOutput:
        if self.dictionary_format is None:
            return output
        return self.dictionary_format.decode(output)

    def encode(self, output: LLMOutput) -> LLMOutput:
        if self.dictionary_format is None:
            return output
        return self.dictionary_format.encode(output)
