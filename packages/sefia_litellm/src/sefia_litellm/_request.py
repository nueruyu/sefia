from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from typing_extensions import final

from sefia.llm import Message
from sefia.llm.json_schema import JsonObject
from sefia.llm.step_decision import DecisionSpec

from ._schema import StructuredDecisionFormat


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
    output: StructuredDecisionFormat | None


def prepare_request(
    *,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    decision_model: DecisionSpec | None,
    client_kwargs: dict[str, Any],
    stream: bool,
) -> PreparedRequest:
    raw_messages = [message.to_dict(exclude_none=True) for message in messages]
    output = (
        StructuredDecisionFormat.from_model(decision_model)
        if decision_model is not None
        else None
    )
    kwargs = client_kwargs.copy()
    if tools:
        kwargs["tools"] = tools
    if output is not None:
        kwargs["response_format"] = _response_format(output)
    if stream:
        kwargs["stream"] = True
    return PreparedRequest(raw_messages, kwargs, output)


def _response_format(output: StructuredDecisionFormat) -> _JsonSchemaResponseFormat:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": output.schema.to_dict(),
            "strict": True,
        },
    }


__all__ = ["PreparedRequest", "prepare_request"]
