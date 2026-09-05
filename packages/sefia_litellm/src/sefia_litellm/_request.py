import json
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from typing_extensions import final

from sefia.llm import Message, ToolCall
from sefia.llm.json_schema import JsonObject
from sefia.llm.step_decision import DecisionSpec, StepTool

from ._schema import StructuredDecisionFormat
from ._schema._data_format import StructuredDataFormat


class _JsonSchemaResponseDefinition(TypedDict):
    name: str
    schema: JsonObject
    strict: bool


class _JsonSchemaResponseFormat(TypedDict):
    type: Literal["json_schema"]
    json_schema: _JsonSchemaResponseDefinition


@final
@dataclass(frozen=True)
class CompletionRequest:
    messages: list[dict[str, Any]]
    api_kwargs: dict[str, Any]
    decision_format: StructuredDecisionFormat | None
    tool_data_formats: dict[str, StructuredDataFormat]


def build_completion_request(
    *,
    messages: list[Message],
    tools: list[StepTool] | None,
    decision_spec: DecisionSpec | None,
    client_kwargs: dict[str, Any],
    stream: bool,
) -> CompletionRequest:
    tool_data_formats = {
        tool.name: StructuredDataFormat.from_tool(tool) for tool in tools or []
    }
    wire_messages = [
        _encode_message(message, tool_data_formats) for message in messages
    ]
    decision_format = (
        StructuredDecisionFormat.from_spec(decision_spec)
        if decision_spec is not None
        else None
    )
    api_kwargs = client_kwargs.copy()
    if tools:
        api_kwargs["tools"] = [
            _encode_tool_definition(tool, tool_data_formats[tool.name])
            for tool in tools
        ]
    if decision_format is not None:
        api_kwargs["response_format"] = _response_format(decision_format)
    if stream:
        api_kwargs["stream"] = True
    return CompletionRequest(
        wire_messages,
        api_kwargs,
        decision_format,
        tool_data_formats,
    )


def _encode_tool_definition(
    tool: StepTool,
    data_format: StructuredDataFormat,
) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "parameters": data_format.schema,
    }
    if tool.description:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _encode_message(
    message: Message,
    tool_data_formats: dict[str, StructuredDataFormat],
) -> dict[str, Any]:
    wire_message: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        wire_message["content"] = message.content
    if message.tool_call_id is not None:
        wire_message["tool_call_id"] = message.tool_call_id
    if message.tool_calls is not None:
        wire_message["tool_calls"] = [
            _encode_tool_call(call, tool_data_formats.get(call.name))
            for call in message.tool_calls
        ]
    return wire_message


def _encode_tool_call(
    call: ToolCall,
    data_format: StructuredDataFormat | None,
) -> dict[str, Any]:
    arguments = (
        data_format.encode(call.arguments)
        if data_format is not None
        else call.arguments
    )
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(
                arguments.tree,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


def _response_format(
    decision_format: StructuredDecisionFormat,
) -> _JsonSchemaResponseFormat:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": decision_format.schema.to_dict(),
            "strict": True,
        },
    }


__all__ = ["CompletionRequest", "build_completion_request"]
