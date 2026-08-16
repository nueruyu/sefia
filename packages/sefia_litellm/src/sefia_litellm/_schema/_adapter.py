from dataclasses import dataclass

from typing_extensions import final

from sefia.llm.schema import (
    JsonObject,
    JsonSchemaDocument,
    JsonValue,
    LLMSchema,
    SchemaPath,
    StructuredValue,
    to_structured_value,
)

from ._decoder import Decoder, DecoderFactory
from ._normalization import CompatibilityValidator, MappingLowerer, SchemaNormalizer


@final
@dataclass
class LiteLLMPreparedSchema:
    wire_schema: JsonSchemaDocument
    _decoder: Decoder

    def decode(self, data: JsonValue) -> StructuredValue:
        decoded = self._decoder.decode(to_structured_value(data))
        if not isinstance(decoded, dict) or set(decoded) != {"payload"}:
            return decoded
        return decoded["payload"]

    def normalize_stream_path(self, path: SchemaPath) -> SchemaPath | None:
        return path[1:] if path and path[0] == "payload" else path


@final
class LiteLLMSchemaAdapter:
    def build(self, logical: LLMSchema) -> LiteLLMPreparedSchema:
        schema, preserved = _compose_envelope(logical)
        SchemaNormalizer(preserved).normalize(schema)
        plan = MappingLowerer(preserved).lower(schema)
        CompatibilityValidator().validate(schema)
        schema["description"] = "The model for the LLM's decision on the next action."
        document = JsonSchemaDocument(schema)
        return LiteLLMPreparedSchema(
            document, DecoderFactory(schema, plan).build(schema)
        )


def _compose_envelope(
    logical: LLMSchema,
) -> tuple[JsonObject, frozenset[SchemaPath]]:
    payload = logical.document.mutable_copy()
    definitions = payload.pop("$defs", None)
    schema: JsonObject = {
        "type": "object",
        "properties": {"payload": payload},
        "required": ["payload"],
        "additionalProperties": False,
    }
    if isinstance(definitions, dict):
        schema["$defs"] = definitions
    preserved = frozenset(
        path if path and path[0] == "$defs" else ("properties", "payload", *path)
        for path in logical.raw_schema_paths
    )
    return schema, preserved
