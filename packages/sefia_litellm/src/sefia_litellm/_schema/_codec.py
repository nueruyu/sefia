from dataclasses import dataclass

from typing_extensions import final

from sefia.llm.json_schema import (
    JsonObject,
    JsonScalar,
    JsonSchemaDocument,
    JsonValue,
    SchemaKeyword,
    SchemaNode,
    SchemaPath,
)
from sefia.llm.llm_output import LLMOutput

from ._normalization import MappingPlan
from ._traversal import matches, resolve

K = SchemaKeyword


@final
@dataclass(frozen=True)
class MappingRestorer:
    schema: JsonObject
    plan: MappingPlan

    def decode(self, output: LLMOutput) -> LLMOutput:
        return _decode(output, self.schema, self.schema, self.plan.mapping_paths, ())


@final
class OutputCodec:
    def __init__(
        self,
        result: MappingRestorer | None,
        tools: dict[str, MappingRestorer | None],
    ) -> None:
        self._result = result
        self._tools = tools

    def decode(self, data: JsonValue) -> LLMOutput:
        output = LLMOutput.from_json(data)
        try:
            envelope = output.to_object()
        except ValueError:
            return output
        if set(envelope) != {"payload"}:
            return output
        return self._decode_decision(envelope["payload"])

    @staticmethod
    def logical_path(path: SchemaPath) -> SchemaPath | None:
        return path[1:] if path and path[0] == "payload" else path

    def _decode_decision(self, output: LLMOutput) -> LLMOutput:
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
        codec = self._tools.get(tool_name) if tool_name is not None else None
        if codec is None or "arguments" not in fields:
            return output
        return LLMOutput.from_object(
            {**fields, "arguments": codec.decode(fields["arguments"])}
        )


@final
@dataclass(frozen=True)
class PreparedOutput:
    wire_schema: JsonSchemaDocument
    codec: OutputCodec

    def decode(self, data: JsonValue) -> LLMOutput:
        return self.codec.decode(data)

    def logical_path(self, path: SchemaPath) -> SchemaPath | None:
        return self.codec.logical_path(path)


def _decode(
    output: LLMOutput,
    schema: JsonObject,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> LLMOutput:
    schema, path = resolve(schema, root, path)
    node = SchemaNode(schema)
    if path in mapping_paths:
        return _decode_mapping(output, node, root, mapping_paths, path)

    alternatives = node.any_of()
    if alternatives:
        for index, alternative in enumerate(alternatives):
            if matches(output.data, alternative.value, root):
                return _decode(
                    output,
                    alternative.value,
                    root,
                    mapping_paths,
                    (*path, K.ANY_OF, index),
                )
        return output

    if node.type == "object":
        return _decode_object(output, node, root, mapping_paths, path)
    if node.type == "array":
        return _decode_array(output, node, root, mapping_paths, path)
    return output


def _decode_object(
    output: LLMOutput,
    node: SchemaNode,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> LLMOutput:
    try:
        fields = output.to_object()
    except ValueError:
        return output
    properties = node.properties()
    return LLMOutput.from_object(
        {
            name: _decode(
                value,
                properties[name].value,
                root,
                mapping_paths,
                (*path, K.PROPERTIES, name),
            )
            if name in properties
            else value
            for name, value in fields.items()
        }
    )


def _decode_array(
    output: LLMOutput,
    node: SchemaNode,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> LLMOutput:
    items = node.items()
    if items is None:
        return output
    try:
        values = output.to_array()
    except ValueError:
        return output
    return LLMOutput.from_array(
        _decode(value, items.value, root, mapping_paths, (*path, K.ITEMS))
        for value in values
    )


def _decode_mapping(
    output: LLMOutput,
    node: SchemaNode,
    root: JsonObject,
    mapping_paths: frozenset[SchemaPath],
    path: SchemaPath,
) -> LLMOutput:
    items = node.items()
    properties = items.properties() if items is not None else {}
    if set(properties) != {"key", "value"}:
        raise ValueError("lowered mapping schema is missing key/value entries")
    entries = output.to_array()
    result: dict[JsonScalar, LLMOutput] = {}
    for entry in entries:
        fields = entry.to_object("mapping entry")
        if set(fields) != {"key", "value"}:
            raise ValueError("mapping entries must contain only key and value")
        key = _decode(
            fields["key"],
            properties["key"].value,
            root,
            mapping_paths,
            (*path, K.ITEMS, K.PROPERTIES, "key"),
        ).to_scalar("mapping key")
        if key in result:
            raise ValueError(f"duplicate mapping key: {key!r}")
        result[key] = _decode(
            fields["value"],
            properties["value"].value,
            root,
            mapping_paths,
            (*path, K.ITEMS, K.PROPERTIES, "value"),
        )
    return LLMOutput.from_mapping(result)


__all__ = ["OutputCodec", "PreparedOutput", "MappingRestorer"]
