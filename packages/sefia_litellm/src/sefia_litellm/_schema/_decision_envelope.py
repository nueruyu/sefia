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
    StepTool,
    TypedToolArguments,
)

from ._fragment import CompiledFragment
from ._policy import (
    GENERATED_SCHEMA_POLICY,
    USER_DEFINED_SCHEMA_POLICY,
    SchemaPolicy,
    prepare_schema,
)

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
        result: CompiledFragment | None,
        tools: dict[str, CompiledFragment],
    ) -> None:
        self._schema = schema
        self._result = result
        self._tools = tools

    @classmethod
    def from_model(cls, model: StepDecisionModel) -> "DecisionEnvelopeFormat":
        result = (
            _compile_fragment(model.result.json_schema, GENERATED_SCHEMA_POLICY)
            if model.result is not None
            else None
        )
        tools = {tool.name: _compile_tool_fragment(tool) for tool in model.tools}
        schema = _build_schema(model.mode, result, tools)
        return cls(JsonSchemaDocument(schema), result, tools)

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
    def logical_path(path: SchemaPath) -> SchemaPath | None:
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
        if self._result is None or "result" not in fields:
            return output
        return LLMOutput.from_object(
            {**fields, "result": self._result.decode(fields["result"])}
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
        fragment = self._tools.get(tool_name) if tool_name is not None else None
        if fragment is None or "arguments" not in fields:
            return output
        return LLMOutput.from_object(
            {**fields, "arguments": fragment.decode(fields["arguments"])}
        )


def _compile_tool_fragment(tool: StepTool) -> CompiledFragment:
    if isinstance(tool.arguments, TypedToolArguments):
        return _compile_fragment(tool.arguments.json_schema, GENERATED_SCHEMA_POLICY)
    return _compile_fragment(tool.arguments.json_schema, USER_DEFINED_SCHEMA_POLICY)


def _compile_fragment(
    document: JsonSchemaDocument, policy: SchemaPolicy
) -> CompiledFragment:
    prepared = prepare_schema(document.mutable_copy(), policy)
    return CompiledFragment(prepared.wire_schema, prepared.mapping)


def _build_schema(
    mode: StepDecisionMode,
    result: CompiledFragment | None,
    tools: dict[str, CompiledFragment],
) -> JsonObject:
    definitions: JsonObject = {}
    registry = DefinitionRegistry(definitions)
    payload = _payload_schema(mode, result, tools, registry)
    root = SchemaNode.object_schema({"payload": payload})
    if definitions:
        root.set_definitions(definitions)
    root.set_description("The model for the LLM's decision on the next action.")
    return root.value


def _payload_schema(
    mode: StepDecisionMode,
    result: CompiledFragment | None,
    tools: dict[str, CompiledFragment],
    registry: DefinitionRegistry,
) -> JsonObject:
    branches: list[JsonObject] = []
    if mode is not StepDecisionMode.RESULT_ONLY:
        branches.append(_tool_calls_schema(tools, registry))
    if mode is not StepDecisionMode.TOOLS_REQUIRED:
        assert result is not None
        schema = JsonSchemaDocument(result.wire_schema).mutable_copy()
        imported = registry.import_schema(schema, namespace="result")
        branches.append(
            _closed_object({"decision": _literal("result"), "result": imported})
        )
    if len(branches) == 1:
        return branches[0]
    return cast(
        JsonObject,
        {
            K.ANY_OF: branches,
            "discriminator": {"propertyName": "decision"},
        },
    )


def _tool_calls_schema(
    tools: dict[str, CompiledFragment], registry: DefinitionRegistry
) -> JsonObject:
    calls: list[JsonObject] = []
    for name, fragment in tools.items():
        schema = JsonSchemaDocument(fragment.wire_schema).mutable_copy()
        imported = registry.import_schema(schema, namespace=name)
        calls.append(_closed_object({"name": _literal(name), "arguments": imported}))
    items: JsonObject = (
        calls[0]
        if len(calls) == 1
        else cast(
            JsonObject,
            {K.ANY_OF: calls, "discriminator": {"propertyName": "name"}},
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
