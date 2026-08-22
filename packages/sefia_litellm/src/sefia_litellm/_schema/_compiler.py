from dataclasses import dataclass

from typing_extensions import final

from sefia.llm.json_schema import JsonSchemaDocument, JsonValue, SchemaPath
from sefia.llm.llm_output import LLMOutput
from sefia.llm.step_decision import StepDecisionModel, StepTool, TypedToolArguments

from ._decision_envelope import DecisionEnvelope
from ._dialect import StructuredOutputDialect
from ._fragment import CompiledFragment
from ._mapping import MappingTransform


@final
@dataclass(frozen=True)
class CompiledOutputSchema:
    wire_schema: JsonSchemaDocument
    envelope: DecisionEnvelope

    def decode(self, data: JsonValue) -> LLMOutput:
        return self.envelope.decode(data)

    def logical_path(self, path: SchemaPath) -> SchemaPath | None:
        return self.envelope.logical_path(path)


@final
class OutputSchemaCompiler:
    def __init__(self) -> None:
        self._dialect = StructuredOutputDialect()

    def compile(self, model: StepDecisionModel) -> CompiledOutputSchema:
        result = (
            self._compile_typed(model.result.json_schema)
            if model.result is not None
            else None
        )
        tools = {tool.name: self._compile_tool(tool) for tool in model.tools}
        envelope = DecisionEnvelope(model.mode, result, tools)
        wire_schema = envelope.build_schema()
        self._dialect.validate(wire_schema)
        return CompiledOutputSchema(JsonSchemaDocument(wire_schema), envelope)

    def _compile_tool(self, tool: StepTool) -> CompiledFragment:
        if isinstance(tool.arguments, TypedToolArguments):
            return self._compile_typed(tool.arguments.json_schema)
        wire_schema = tool.arguments.json_schema.mutable_copy()
        self._dialect.validate(wire_schema)
        return CompiledFragment(wire_schema)

    def _compile_typed(self, document: JsonSchemaDocument) -> CompiledFragment:
        wire_schema = document.mutable_copy()
        self._dialect.adapt(wire_schema)
        mapping = MappingTransform.lower(wire_schema)
        self._dialect.validate(wire_schema)
        return CompiledFragment(wire_schema, mapping)
