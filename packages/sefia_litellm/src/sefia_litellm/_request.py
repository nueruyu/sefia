import json
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from typing_extensions import final

from sefia.llm import Message, ToolCall
from sefia.llm.json_schema import JsonObject
from sefia.llm.step_decision import DecisionSpec, StepTool

from ._schema import (
    StructuredDecisionFormat,
    StructuredValueFormat,
    tool_arguments_format,
)


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
    kwargs: dict[str, Any]
    decision_format: StructuredDecisionFormat | None
    tool_argument_formats: dict[str, StructuredValueFormat]


def build_completion_request(
    *,
    messages: list[Message],
    tools: list[StepTool] | None,
    decision_model: DecisionSpec | None,
    client_kwargs: dict[str, Any],
    stream: bool,
) -> CompletionRequest:
    tool_argument_formats = {
        tool.name: tool_arguments_format(tool) for tool in tools or []
    }
    raw_messages = [_message(message, tool_argument_formats) for message in messages]
    decision_format = (
        StructuredDecisionFormat.from_model(decision_model)
        if decision_model is not None
        else None
    )
    kwargs = client_kwargs.copy()
    if tools:
        kwargs["tools"] = [
            _tool_definition(tool, tool_argument_formats[tool.name]) for tool in tools
        ]
    if decision_format is not None:
        kwargs["response_format"] = _response_format(decision_format)
    if stream:
        kwargs["stream"] = True
    return CompletionRequest(
        raw_messages,
        kwargs,
        decision_format,
        tool_argument_formats,
    )


def _tool_definition(
    tool: StepTool,
    value_format: StructuredValueFormat,
) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "parameters": value_format.schema,
    }
    if tool.description:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _message(
    message: Message,
    tool_argument_formats: dict[str, StructuredValueFormat],
) -> dict[str, Any]:
    result: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        result["content"] = message.content
    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id
    if message.tool_calls is not None:
        result["tool_calls"] = [
            _tool_call(call, tool_argument_formats.get(call.name))
            for call in message.tool_calls
        ]
    return result


def _tool_call(
    call: ToolCall,
    value_format: StructuredValueFormat | None,
) -> dict[str, Any]:
    arguments = (
        value_format.encode(call.arguments)
        if value_format is not None
        else call.arguments
    )
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(
                arguments.data,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


def _response_format(output: StructuredDecisionFormat) -> _JsonSchemaResponseFormat:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": output.schema.to_dict(),
            "strict": True,
        },
    }


__all__ = ["CompletionRequest", "build_completion_request"]
