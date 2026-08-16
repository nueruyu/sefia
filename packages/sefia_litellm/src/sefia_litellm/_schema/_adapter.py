from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from typing_extensions import final

from sefia.llm.schema import LLMSchema, SchemaPath

from ._decoder import Decoder, DecoderFactory
from ._normalization import CompatibilityValidator, MappingLowerer, SchemaNormalizer
from ._traversal import walk_with_paths


@final
@dataclass
class LiteLLMPreparedSchema:
    _schema: dict[str, Any]
    _decoder: Decoder

    @property
    def schema(self) -> dict[str, Any]:
        return deepcopy(self._schema)

    def decode(self, data: object) -> object:
        decoded = self._decoder.decode(data)
        if not isinstance(decoded, dict):
            return decoded
        decoded_map = cast(dict[str, Any], decoded)
        if set(decoded_map) == {"payload"}:
            return decoded_map["payload"]
        return decoded_map

    def normalize_stream_path(self, path: SchemaPath) -> SchemaPath | None:
        return path[1:] if path and path[0] == "payload" else path


@final
class LiteLLMSchemaAdapter:
    def build(self, logical: LLMSchema) -> LiteLLMPreparedSchema:
        schema, preserved = _compose_envelope(logical)
        SchemaNormalizer(preserved).normalize(schema)
        mapping_ids = MappingLowerer(preserved).lower(schema)
        CompatibilityValidator().validate(schema)
        schema["description"] = "The model for the LLM's decision on the next action."
        return LiteLLMPreparedSchema(
            schema, DecoderFactory(schema, mapping_ids).build(schema)
        )


def _compose_envelope(logical: LLMSchema) -> tuple[dict[str, Any], set[int]]:
    payload = deepcopy(logical.schema)
    preserved = {
        id(node)
        for path, node in walk_with_paths(payload)
        if path in logical.raw_schema_paths
    }
    definitions = payload.pop("$defs", None)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"payload": payload},
        "required": ["payload"],
        "additionalProperties": False,
    }
    if isinstance(definitions, dict):
        schema["$defs"] = definitions
    return schema, preserved
