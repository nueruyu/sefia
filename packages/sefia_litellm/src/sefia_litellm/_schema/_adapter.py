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
from sefia.llm.step_decision import (
    StepDecisionMode,
    StepDecisionModel,
    StepTool,
    TypedToolArguments,
)
from sefia.llm.structured_output import StructuredValue, to_structured_value

from ._decoder import Decoder, DecoderFactory
from ._normalization import CompatibilityValidator, MappingLowerer, SchemaNormalizer

K = SchemaKeyword


@final
@dataclass
class LiteLLMPreparedSchema:
    wire_schema: JsonSchemaDocument
    _decoder: Decoder

    def decode(self, data: JsonValue) -> StructuredValue:
        value = to_structured_value(data)
        if not isinstance(value, dict) or set(value) != {"payload"}:
            return value
        return self._decoder.decode(value["payload"])

    def normalize_stream_path(self, path: SchemaPath) -> SchemaPath | None:
        return path[1:] if path and path[0] == "payload" else path


@final
class _IdentityDecoder:
    def decode(self, data: StructuredValue) -> StructuredValue:
        return data


@final
class _StepDecisionDecoder:
    def __init__(self, result: Decoder | None, tools: dict[str, Decoder]):
        self._result = result
        self._tools = tools

    def decode(self, data: StructuredValue) -> StructuredValue:
        if not isinstance(data, dict):
            return data
        decision = data.get("decision")
        if decision == "result" and self._result is not None and "result" in data:
            return {**data, "result": self._result.decode(data["result"])}
        tool_calls = data.get("tool_calls")
        if decision != "tool_calls" or not isinstance(tool_calls, list):
            return data
        calls: list[StructuredValue] = []
        for value in tool_calls:
            if not isinstance(value, dict):
                calls.append(value)
                continue
            name = value.get("name")
            decoder = self._tools.get(name) if isinstance(name, str) else None
            if decoder is None or "arguments" not in value:
                calls.append(value)
                continue
            calls.append({**value, "arguments": decoder.decode(value["arguments"])})
        return {**data, "tool_calls": calls}


@dataclass(frozen=True)
class _PreparedFragment:
    schema: JsonObject
    decoder: Decoder


@final
class LiteLLMStructuredOutputAdapter:
    def build(self, model: StepDecisionModel) -> LiteLLMPreparedSchema:
        result = (
            _prepare_typed(model.result.json_schema)
            if model.result is not None
            else None
        )
        tools = {tool.name: _prepare_tool(tool) for tool in model.tools}
        definitions: JsonObject = {}
        registry = DefinitionRegistry(definitions)
        payload = _compose_decision(model.mode, result, tools, registry)
        root = SchemaNode.object_schema({"payload": payload})
        if definitions:
            root.set_definitions(definitions)
        root.set_description("The model for the LLM's decision on the next action.")
        CompatibilityValidator().validate(root.value)
        return LiteLLMPreparedSchema(
            JsonSchemaDocument(root.value),
            _StepDecisionDecoder(
                result.decoder if result is not None else None,
                {name: fragment.decoder for name, fragment in tools.items()},
            ),
        )


def _prepare_tool(tool: StepTool) -> _PreparedFragment:
    if isinstance(tool.arguments, TypedToolArguments):
        return _prepare_typed(tool.arguments.json_schema)
    schema = tool.arguments.json_schema.mutable_copy()
    CompatibilityValidator().validate(schema)
    return _PreparedFragment(schema, _IdentityDecoder())


def _prepare_typed(document: JsonSchemaDocument) -> _PreparedFragment:
    schema = document.mutable_copy()
    SchemaNormalizer().normalize(schema)
    plan = MappingLowerer().lower(schema)
    CompatibilityValidator().validate(schema)
    return _PreparedFragment(schema, DecoderFactory(schema, plan).build(schema))


def _compose_decision(
    mode: StepDecisionMode,
    result: _PreparedFragment | None,
    tools: dict[str, _PreparedFragment],
    registry: DefinitionRegistry,
) -> JsonObject:
    branches: list[JsonObject] = []
    if mode is not StepDecisionMode.RESULT_ONLY:
        branches.append(_tool_calls_branch(tools, registry))
    if mode is not StepDecisionMode.TOOLS_REQUIRED:
        assert result is not None
        imported = registry.import_schema(result.schema, namespace="result")
        branches.append(
            _closed_object({"decision": _literal("result"), "result": imported.schema})
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


def _tool_calls_branch(
    tools: dict[str, _PreparedFragment], registry: DefinitionRegistry
) -> JsonObject:
    calls: list[JsonObject] = []
    for name, fragment in tools.items():
        imported = registry.import_schema(fragment.schema, namespace=name)
        calls.append(
            _closed_object({"name": _literal(name), "arguments": imported.schema})
        )
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
