from dataclasses import dataclass

from sefia.json_schema import (
    DefinitionRegistry,
    JsonSchemaDocument,
    SchemaKeyword,
    SchemaNode,
)
from sefia.llm.json import JsonCompatible, JsonSnapshot
from sefia.llm.step_decision import (
    DecisionSpec,
    StepDecisionMode,
    StepTool,
    ToolSchemaSource,
)
from sefia.llm.streaming import OutputStreamEvent, Scalar, StringDelta, StringEnd
from typing_extensions import final

from ._types import SchemaObject
from ._provider_format import ProviderJsonFormat

K = SchemaKeyword
_PAYLOAD_FIELD = "payload"


@final
@dataclass(frozen=True)
class _ToolProviderFormat:
    arguments: ProviderJsonFormat
    description: str | None


@final
class StructuredDecisionFormat:
    def __init__(
        self,
        schema: JsonSchemaDocument,
        result_provider_format: ProviderJsonFormat | None,
        tool_provider_formats: dict[str, _ToolProviderFormat],
    ) -> None:
        self._schema = schema
        self._result_provider_format = result_provider_format
        self._tool_provider_formats = tool_provider_formats

    @classmethod
    def from_spec(cls, spec: DecisionSpec) -> "StructuredDecisionFormat":
        result_provider_format = (
            ProviderJsonFormat.from_generated_schema(spec.result.schema)
            if spec.result is not None
            else None
        )
        tool_provider_formats = {
            tool.name: _ToolProviderFormat(
                arguments=_tool_provider_format(tool),
                description=tool.description,
            )
            for tool in spec.tools
        }
        schema = _build_schema(spec.mode, result_provider_format, tool_provider_formats)
        return cls(
            JsonSchemaDocument(schema), result_provider_format, tool_provider_formats
        )

    @property
    def schema(self) -> JsonSchemaDocument:
        return self._schema

    def decode_json(self, text: str) -> JsonSnapshot:
        return self._decode(JsonSnapshot.parse_json(text))

    def decode(self, data: JsonCompatible) -> JsonSnapshot:
        return self._decode(JsonSnapshot.capture(data))

    def _decode(self, data: JsonSnapshot) -> JsonSnapshot:
        envelope = data.as_object("structured decision envelope")
        if set(envelope) != {_PAYLOAD_FIELD}:
            raise ValueError(
                "structured decision envelope must contain only the payload field"
            )
        data = envelope[_PAYLOAD_FIELD]

        try:
            fields = data.as_object()
        except ValueError:
            return data
        decision = fields.get("decision")
        try:
            decision_name = decision.as_string() if decision is not None else None
        except ValueError:
            return data
        if decision_name == "result":
            return self._decode_result(data, fields)
        if decision_name == "tool_calls":
            return self._decode_tool_calls(data, fields)
        return data

    def _decode_result(
        self, data: JsonSnapshot, fields: dict[str, JsonSnapshot]
    ) -> JsonSnapshot:
        if self._result_provider_format is None or "result" not in fields:
            return data
        return JsonSnapshot.from_object(
            {**fields, "result": self._result_provider_format.decode(fields["result"])}
        )

    def _decode_tool_calls(
        self, data: JsonSnapshot, fields: dict[str, JsonSnapshot]
    ) -> JsonSnapshot:
        tool_calls = fields.get("tool_calls")
        if tool_calls is None:
            return data
        try:
            calls = tool_calls.as_array()
        except ValueError:
            return data
        return JsonSnapshot.from_object(
            {
                **fields,
                "tool_calls": JsonSnapshot.from_array(
                    self._decode_tool_call(call) for call in calls
                ),
            }
        )

    def _decode_tool_call(self, data: JsonSnapshot) -> JsonSnapshot:
        try:
            fields = data.as_object()
        except ValueError:
            return data
        name = fields.get("name")
        try:
            tool_name = name.as_string() if name is not None else None
        except ValueError:
            return data
        tool_provider_format = (
            self._tool_provider_formats.get(tool_name)
            if tool_name is not None
            else None
        )
        if tool_provider_format is None or "arguments" not in fields:
            return data
        return JsonSnapshot.from_object(
            {
                **fields,
                "arguments": tool_provider_format.arguments.decode(fields["arguments"]),
            }
        )

    def decode_stream_event(self, event: OutputStreamEvent) -> OutputStreamEvent | None:
        if not event.path or event.path[0] != _PAYLOAD_FIELD:
            return None
        path = event.path[1:]
        if isinstance(event, StringDelta):
            return StringDelta(path, event.text)
        if isinstance(event, StringEnd):
            return StringEnd(path, event.value)
        return Scalar(path, event.value)


def _tool_provider_format(tool: StepTool) -> ProviderJsonFormat:
    if tool.schema_source is ToolSchemaSource.GENERATED:
        return ProviderJsonFormat.from_generated_schema(tool.arguments)
    return ProviderJsonFormat.from_user_schema(tool.arguments)


def _build_schema(
    mode: StepDecisionMode,
    result_provider_format: ProviderJsonFormat | None,
    tool_provider_formats: dict[str, _ToolProviderFormat],
) -> SchemaObject:
    definitions: SchemaObject = {}
    registry = DefinitionRegistry(definitions)
    decision = _decision_schema(
        mode,
        result_provider_format,
        tool_provider_formats,
        registry,
    )
    root = SchemaNode.object_schema({_PAYLOAD_FIELD: decision})
    if definitions:
        root.set_definitions(definitions)
    return root.value


def _decision_schema(
    mode: StepDecisionMode,
    result_provider_format: ProviderJsonFormat | None,
    tool_provider_formats: dict[str, _ToolProviderFormat],
    registry: DefinitionRegistry,
) -> SchemaObject:
    branches: list[SchemaObject] = []
    if mode is not StepDecisionMode.RESULT_ONLY:
        branches.append(
            _tool_calls_schema(
                tool_provider_formats,
                registry,
            )
        )
    if mode is not StepDecisionMode.TOOLS_REQUIRED:
        assert result_provider_format is not None
        imported = registry.import_schema(
            result_provider_format.schema, namespace="result"
        )
        branches.append(
            _closed_object({"decision": _literal("result"), "result": imported})
        )
    if len(branches) == 1:
        return branches[0]
    return _branch_union(branches)


def _tool_calls_schema(
    tool_provider_formats: dict[str, _ToolProviderFormat],
    registry: DefinitionRegistry,
) -> SchemaObject:
    calls: list[SchemaObject] = []
    for index, (name, tool_provider_format) in enumerate(tool_provider_formats.items()):
        imported = registry.import_schema(
            tool_provider_format.arguments.schema,
            namespace=f"tool_{index}",
        )
        call = _closed_object({"name": _literal(name), "arguments": imported})
        if tool_provider_format.description:
            call[K.DESCRIPTION] = tool_provider_format.description
        calls.append(call)
    items: SchemaObject = calls[0] if len(calls) == 1 else _branch_union(calls)
    return _closed_object(
        {
            "decision": _literal("tool_calls"),
            "tool_calls": {K.TYPE: "array", K.ITEMS: items, K.MIN_ITEMS: 1},
        }
    )


def _closed_object(properties: SchemaObject) -> SchemaObject:
    return {
        K.TYPE: "object",
        K.PROPERTIES: properties,
        K.REQUIRED: list(properties),
        K.ADDITIONAL_PROPERTIES: False,
    }


def _branch_union(branches: list[SchemaObject]) -> SchemaObject:
    """Build a provider-compatible union of const-disjoint schema branches."""
    # Anthropic rejects OpenAPI's discriminator in native structured output.
    return {K.ANY_OF: [*branches]}


def _literal(value: str) -> SchemaObject:
    return {K.TYPE: "string", K.CONST: value}
