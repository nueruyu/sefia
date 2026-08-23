from dataclasses import dataclass
from typing import cast

from typing_extensions import final

from sefia.llm.json_schema import (
    DefinitionRegistry,
    JsonObject,
    JsonSchemaDocument,
    JsonValue,
    SchemaKeyword,
    SchemaNode,
    SchemaPath,
)
from sefia.llm.llm_output import LLMOutput
from sefia.llm.step_decision import (
    StepDecisionMode,
    StepDecisionModel,
)

from ._value_format import StructuredValueFormat

K = SchemaKeyword


@final
@dataclass(frozen=True)
class DecisionEnvelope:
    payload: LLMOutput

    @classmethod
    def from_output(cls, output: LLMOutput) -> "DecisionEnvelope":
        fields = output.to_object("decision envelope")
        if set(fields) != {"payload"}:
            raise ValueError("decision envelope must contain exactly one payload")
        return cls(fields["payload"])


@final
class DecisionEnvelopeFormat:
    def __init__(
        self,
        schema: JsonSchemaDocument,
        result_format: StructuredValueFormat | None,
        tool_argument_formats: dict[str, StructuredValueFormat],
    ) -> None:
        self._schema = schema
        self._result_format = result_format
        self._tool_argument_formats = tool_argument_formats

    @classmethod
    def from_model(cls, model: StepDecisionModel) -> "DecisionEnvelopeFormat":
        result_format = (
            StructuredValueFormat.from_generated_schema(model.result.schema)
            if model.result is not None
            else None
        )
        tool_argument_formats = {
            tool.name: StructuredValueFormat.from_tool(tool) for tool in model.tools
        }
        tool_descriptions = {tool.name: tool.description for tool in model.tools}
        schema = _build_schema(
            model.mode,
            result_format,
            tool_argument_formats,
            tool_descriptions,
        )
        return cls(JsonSchemaDocument(schema), result_format, tool_argument_formats)

    @property
    def schema(self) -> JsonSchemaDocument:
        return self._schema

    def decode_json(self, text: str) -> LLMOutput:
        return self._decode(LLMOutput.parse_json(text))

    def decode(self, data: JsonValue) -> LLMOutput:
        return self._decode(LLMOutput.from_json(data))

    def _decode(self, output: LLMOutput) -> LLMOutput:
        try:
            envelope = DecisionEnvelope.from_output(output)
        except ValueError:
            return output
        return self._decode_payload(envelope.payload)

    @staticmethod
    def to_payload_path(path: SchemaPath) -> SchemaPath | None:
        return path[1:] if path and path[0] == "payload" else path

    def _decode_payload(self, output: LLMOutput) -> LLMOutput:
        try:
            fields = output.to_object()
        except ValueError:
            return output
        decision = fields.get("decision")
        try:
            decision_name = decision.to_string() if decision is not None else None
        except ValueError:
            return output
        if decision_name == "result":
            return self._decode_result(output, fields)
        if decision_name == "tool_calls":
            return self._decode_tool_calls(output, fields)
        return output

    def _decode_result(
        self, output: LLMOutput, fields: dict[str, LLMOutput]
    ) -> LLMOutput:
        if self._result_format is None or "result" not in fields:
            return output
        return LLMOutput.from_object(
            {**fields, "result": self._result_format.decode(fields["result"])}
        )

    def _decode_tool_calls(
        self, output: LLMOutput, fields: dict[str, LLMOutput]
    ) -> LLMOutput:
        tool_calls = fields.get("tool_calls")
        if tool_calls is None:
            return output
        try:
            calls = tool_calls.to_array()
        except ValueError:
            return output
        return LLMOutput.from_object(
            {
                **fields,
                "tool_calls": LLMOutput.from_array(
                    self._decode_tool_call(call) for call in calls
                ),
            }
        )

    def _decode_tool_call(self, output: LLMOutput) -> LLMOutput:
        try:
            fields = output.to_object()
        except ValueError:
            return output
        name = fields.get("name")
        try:
            tool_name = name.to_string() if name is not None else None
        except ValueError:
            return output
        value_format = (
            self._tool_argument_formats.get(tool_name)
            if tool_name is not None
            else None
        )
        if value_format is None or "arguments" not in fields:
            return output
        return LLMOutput.from_object(
            {**fields, "arguments": value_format.decode(fields["arguments"])}
        )


def _build_schema(
    mode: StepDecisionMode,
    result_format: StructuredValueFormat | None,
    tool_argument_formats: dict[str, StructuredValueFormat],
    tool_descriptions: dict[str, str],
) -> JsonObject:
    definitions: JsonObject = {}
    registry = DefinitionRegistry(definitions)
    payload = _payload_schema(
        mode,
        result_format,
        tool_argument_formats,
        tool_descriptions,
        registry,
    )
    root = SchemaNode.object_schema({"payload": payload})
    if definitions:
        root.set_definitions(definitions)
    root.set_description("The model for the LLM's decision on the next action.")
    return root.value


def _payload_schema(
    mode: StepDecisionMode,
    result_format: StructuredValueFormat | None,
    tool_argument_formats: dict[str, StructuredValueFormat],
    tool_descriptions: dict[str, str],
    registry: DefinitionRegistry,
) -> JsonObject:
    branches: list[JsonObject] = []
    if mode is not StepDecisionMode.RESULT_ONLY:
        branches.append(
            _tool_calls_schema(
                tool_argument_formats,
                tool_descriptions,
                registry,
            )
        )
    if mode is not StepDecisionMode.TOOLS_REQUIRED:
        assert result_format is not None
        imported = registry.import_schema(result_format.schema, namespace="result")
        branches.append(
            _closed_object({"decision": _literal("result"), "result": imported})
        )
    if len(branches) == 1:
        return branches[0]
    return cast(
        JsonObject,
        {K.ANY_OF: branches},
    )


def _tool_calls_schema(
    tool_argument_formats: dict[str, StructuredValueFormat],
    tool_descriptions: dict[str, str],
    registry: DefinitionRegistry,
) -> JsonObject:
    calls: list[JsonObject] = []
    for index, (name, value_format) in enumerate(tool_argument_formats.items()):
        imported = registry.import_schema(
            value_format.schema,
            namespace=f"tool_{index}",
        )
        call = _closed_object({"name": _literal(name), "arguments": imported})
        description = tool_descriptions[name]
        if description:
            call[K.DESCRIPTION] = description
        calls.append(call)
    items: JsonObject = (
        calls[0]
        if len(calls) == 1
        else cast(
            JsonObject,
            {K.ANY_OF: calls},
        )
    )
    return _closed_object(
        {
            "decision": _literal("tool_calls"),
            "tool_calls": {K.TYPE: "array", K.ITEMS: items, K.MIN_ITEMS: 1},
        }
    )


def _closed_object(properties: JsonObject) -> JsonObject:
    return {
        K.TYPE: "object",
        K.PROPERTIES: properties,
        K.REQUIRED: list(properties),
        K.ADDITIONAL_PROPERTIES: False,
    }


def _literal(value: str) -> JsonObject:
    return {K.TYPE: "string", K.CONST: value}
