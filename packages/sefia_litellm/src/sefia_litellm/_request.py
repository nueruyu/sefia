import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from typing_extensions import final

from sefia.llm import Message
from sefia.llm.json_schema import JsonObject
from sefia.llm.step_decision import StepDecisionModel
from sefia.llm.transports import ToolDefinition

from ._schema import DecisionEnvelopeFormat

_SCHEMA_PROMPT = """
### Response Format
Return exactly one raw JSON object matching this schema. Do not include prose,
markdown, or code fences.

{schema}
"""


class _JsonSchemaResponseDefinition(TypedDict):
    name: str
    schema: JsonObject
    strict: bool


class _JsonSchemaResponseFormat(TypedDict):
    type: Literal["json_schema"]
    json_schema: _JsonSchemaResponseDefinition


@final
@dataclass(frozen=True)
class PreparedRequest:
    messages: list[dict[str, Any]]
    kwargs: dict[str, Any]
    output: DecisionEnvelopeFormat | None


def prepare_request(
    *,
    model: str,
    messages: list[Message],
    tools: list[ToolDefinition] | None,
    decision_model: StepDecisionModel | None,
    client_kwargs: dict[str, Any],
    native_structured_output: bool | None,
    supports_response_schema: Callable[..., bool],
    stream: bool,
) -> PreparedRequest:
    raw_messages = [_message(message) for message in messages]
    output = (
        DecisionEnvelopeFormat.from_model(decision_model)
        if decision_model is not None
        else None
    )
    kwargs = client_kwargs.copy()
    if tools:
        kwargs["tools"] = [_tool_definition(tool) for tool in tools]
    if output is not None:
        native = (
            native_structured_output
            if native_structured_output is not None
            else supports_response_schema(model=model)
        )
        if native:
            kwargs["response_format"] = _response_format(output)
        else:
            raw_messages = _with_schema_instruction(
                raw_messages, output.schema.to_dict()
            )
    if stream:
        kwargs["stream"] = True
    return PreparedRequest(raw_messages, kwargs, output)


def _message(message: Message) -> dict[str, Any]:
    raw = message.to_dict(exclude_none=True)
    if message.tool_calls is not None:
        raw["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    return raw


def _tool_definition(tool: ToolDefinition) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "parameters": tool.parameters,
    }
    if tool.description:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _response_format(output: DecisionEnvelopeFormat) -> _JsonSchemaResponseFormat:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": output.schema.to_dict(),
            "strict": True,
        },
    }


def _with_schema_instruction(
    messages: list[dict[str, Any]], schema: JsonObject
) -> list[dict[str, Any]]:
    instruction = _SCHEMA_PROMPT.format(
        schema=json.dumps(schema, indent=2, ensure_ascii=False)
    ).strip()
    result = [message.copy() for message in messages]
    for message in result:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = f"{content}\n\n{instruction}"
            return result
    result.insert(0, {"role": "system", "content": instruction})
    return result


__all__ = ["PreparedRequest", "prepare_request"]
