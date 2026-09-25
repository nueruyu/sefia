from dataclasses import dataclass

from sefia.json_schema import JsonSchemaDocument
from sefia.llm.json import JsonSnapshot
from typing_extensions import final

from ._json import JsonObject
from ._policy import (
    GENERATED_SCHEMA_POLICY,
    USER_DEFINED_SCHEMA_POLICY,
    SchemaPolicy,
    prepare_schema,
)
from ._uniform_dictionary import UniformDictionaryFormat


@final
@dataclass(frozen=True)
class JsonWireFormat:
    schema: JsonObject
    dictionary_format: UniformDictionaryFormat | None

    @property
    def transforms_data(self) -> bool:
        return self.dictionary_format is not None

    @classmethod
    def from_generated_schema(cls, document: JsonSchemaDocument) -> "JsonWireFormat":
        return cls._from_schema(document, GENERATED_SCHEMA_POLICY)

    @classmethod
    def from_user_schema(cls, document: JsonSchemaDocument) -> "JsonWireFormat":
        return cls._from_schema(document, USER_DEFINED_SCHEMA_POLICY)

    @classmethod
    def _from_schema(
        cls, document: JsonSchemaDocument, policy: SchemaPolicy
    ) -> "JsonWireFormat":
        prepared = prepare_schema(document.mutable_copy(), policy)
        return cls(prepared.wire_schema, prepared.dictionary_format)

    def decode(self, data: JsonSnapshot) -> JsonSnapshot:
        if self.dictionary_format is None:
            return data
        return self.dictionary_format.decode(data)

    def encode(self, data: JsonSnapshot) -> JsonSnapshot:
        if self.dictionary_format is None:
            return data
        return self.dictionary_format.encode(data)
