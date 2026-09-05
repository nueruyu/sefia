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
)
from sefia.llm.structured_data import StructuredData
from sefia.llm.step_decision import (
    DecisionSpec,
    StepDecisionMode,
    StepTool,
    ToolSchemaSource,
)

from ._data_format import StructuredDataFormat

K = SchemaKeyword


@final
@dataclass(frozen=True)
class _ToolFormat:
    arguments: StructuredDataFormat
    description: str | None


@final
class StructuredDecisionFormat:
    def __init__(
        self,
        schema: JsonSchemaDocument,
        result_format: StructuredDataFormat | None,
        tool_formats: dict[str, _ToolFormat],
    ) -> None:
        self._schema = schema
        self._result_format = result_format
        self._tool_formats = tool_formats

    @classmethod
    def from_spec(cls, spec: DecisionSpec) -> "StructuredDecisionFormat":
        result_format = (
            StructuredDataFormat.from_generated_schema(spec.result.schema)
            if spec.result is not None
            else None
        )
        tool_formats = {
            tool.name: _ToolFormat(
                arguments=_tool_data_format(tool),
                description=tool.description,
            )
            for tool in spec.tools
        }
        schema = _build_schema(spec.mode, result_format, tool_formats)
        return cls(JsonSchemaDocument(schema), result_format, tool_formats)

    @property
    def schema(self) -> JsonSchemaDocument:
        return self._schema

    def decode_json(self, text: str) -> StructuredData:
        return self._decode(StructuredData.parse_json(text))

    def decode(self, data: JsonValue) -> StructuredData:
        return self._decode(StructuredData.from_json(data))

    def _decode(self, data: StructuredData) -> StructuredData:
        try:
            fields = data.to_object()
        except ValueError:
            return data
        decision = fields.get("decision")
        try:
            decision_name = decision.to_string() if decision is not None else None
        except ValueError:
            return data
        if decision_name == "result":
            return self._decode_result(data, fields)
        if decision_name == "tool_calls":
            return self._decode_tool_calls(data, fields)
        return data

    def _decode_result(
        self, data: StructuredData, fields: dict[str, StructuredData]
    ) -> StructuredData:
        if self._result_format is None or "result" not in fields:
            return data
        return StructuredData.from_object(
            {**fields, "result": self._result_format.decode(fields["result"])}
        )

    def _decode_tool_calls(
        self, data: StructuredData, fields: dict[str, StructuredData]
    ) -> StructuredData:
        tool_calls = fields.get("tool_calls")
        if tool_calls is None:
            return data
        try:
            calls = tool_calls.to_array()
        except ValueError:
            return data
        return StructuredData.from_object(
            {
                **fields,
                "tool_calls": StructuredData.from_array(
                    self._decode_tool_call(call) for call in calls
                ),
            }
        )

    def _decode_tool_call(self, data: StructuredData) -> StructuredData:
        try:
            fields = data.to_object()
        except ValueError:
            return data
        name = fields.get("name")
        try:
            tool_name = name.to_string() if name is not None else None
        except ValueError:
            return data
        tool_format = (
            self._tool_formats.get(tool_name) if tool_name is not None else None
        )
        if tool_format is None or "arguments" not in fields:
            return data
        return StructuredData.from_object(
            {
                **fields,
                "arguments": tool_format.arguments.decode(fields["arguments"]),
            }
        )


def _tool_data_format(tool: StepTool) -> StructuredDataFormat:
    if tool.schema_source is ToolSchemaSource.GENERATED:
        return StructuredDataFormat.from_generated_schema(tool.arguments)
    return StructuredDataFormat.from_user_schema(tool.arguments)


def _build_schema(
    mode: StepDecisionMode,
    result_format: StructuredDataFormat | None,
    tool_formats: dict[str, _ToolFormat],
) -> JsonObject:
    definitions: JsonObject = {}
    registry = DefinitionRegistry(definitions)
    root = SchemaNode(
        _decision_schema(
            mode,
            result_format,
            tool_formats,
            registry,
        )
    )
    if definitions:
        root.set_definitions(definitions)
    return root.value


def _decision_schema(
    mode: StepDecisionMode,
    result_format: StructuredDataFormat | None,
    tool_formats: dict[str, _ToolFormat],
    registry: DefinitionRegistry,
) -> JsonObject:
    branches: list[JsonObject] = []
    if mode is not StepDecisionMode.RESULT_ONLY:
        branches.append(
            _tool_calls_schema(
                tool_formats,
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
    return _branch_union(branches)


def _tool_calls_schema(
    tool_formats: dict[str, _ToolFormat],
    registry: DefinitionRegistry,
) -> JsonObject:
    calls: list[JsonObject] = []
    for index, (name, tool_format) in enumerate(tool_formats.items()):
        imported = registry.import_schema(
            tool_format.arguments.schema,
            namespace=f"tool_{index}",
        )
        call = _closed_object({"name": _literal(name), "arguments": imported})
        if tool_format.description:
            call[K.DESCRIPTION] = tool_format.description
        calls.append(call)
    items: JsonObject = calls[0] if len(calls) == 1 else _branch_union(calls)
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


def _branch_union(branches: list[JsonObject]) -> JsonObject:
    """Build a provider-compatible union of const-disjoint schema branches."""
    # Anthropic rejects OpenAPI's discriminator in native structured output.
    return cast(JsonObject, {K.ANY_OF: branches})


def _literal(value: str) -> JsonObject:
    return {K.TYPE: "string", K.CONST: value}
