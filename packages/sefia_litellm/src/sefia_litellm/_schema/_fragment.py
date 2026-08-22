from dataclasses import dataclass

from sefia.llm.json_schema import JsonObject
from sefia.llm.llm_output import LLMOutput

from ._mapping import MappingTransform


@dataclass(frozen=True)
class CompiledFragment:
    wire_schema: JsonObject
    mapping: MappingTransform | None = None

    def decode(self, output: LLMOutput) -> LLMOutput:
        return self.mapping.restore(output) if self.mapping is not None else output
