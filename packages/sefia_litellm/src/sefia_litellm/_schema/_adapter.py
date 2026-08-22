from dataclasses import dataclass
from typing import cast

from typing_extensions import final

from sefia.llm.json_schema import (
    DefinitionRegistry,
    JsonObject,
    JsonSchemaDocument,
    SchemaKeyword,
    SchemaNode,
)
from sefia.llm.step_decision import (
    StepDecisionMode,
    StepDecisionModel,
    StepTool,
    TypedToolArguments,
)

from ._codec import OutputCodec, PreparedOutput, MappingRestorer
from ._normalization import CompatibilityValidator, MappingLowerer, SchemaNormalizer

K = SchemaKeyword


@dataclass(frozen=True)
class _PreparedFragment:
    schema: JsonObject
    restorer: MappingRestorer | None


@final
class LiteLLMStructuredOutputAdapter:
    def build(self, model: StepDecisionModel) -> PreparedOutput:
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
        return PreparedOutput(
            JsonSchemaDocument(root.value),
            OutputCodec(
                result.restorer if result is not None else None,
                {name: fragment.restorer for name, fragment in tools.items()},
            ),
        )


def _prepare_tool(tool: StepTool) -> _PreparedFragment:
    if isinstance(tool.arguments, TypedToolArguments):
        return _prepare_typed(tool.arguments.json_schema)
    schema = tool.arguments.json_schema.mutable_copy()
    CompatibilityValidator().validate(schema)
    return _PreparedFragment(schema, None)


def _prepare_typed(document: JsonSchemaDocument) -> _PreparedFragment:
    schema = document.mutable_copy()
    SchemaNormalizer().normalize(schema)
    plan = MappingLowerer().lower(schema)
    CompatibilityValidator().validate(schema)
    restoration_schema = JsonSchemaDocument(schema).mutable_copy()
    return _PreparedFragment(schema, MappingRestorer(restoration_schema, plan))


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


def _tool_calls_branch(
    tools: dict[str, _PreparedFragment], registry: DefinitionRegistry
) -> JsonObject:
    calls: list[JsonObject] = []
    for name, fragment in tools.items():
        imported = registry.import_schema(fragment.schema, namespace=name)
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
