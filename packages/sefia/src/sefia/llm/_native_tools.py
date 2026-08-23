import json
from dataclasses import dataclass
from typing import Any, cast

from ..exceptions import UnknownToolDecisionError
from ..inference import ResultDecision, StepDecision, ToolCallRequest, ToolCallsDecision
from ._messages import ToolCall
from .json_schema import JsonValue
from .llm_output import LLMOutput
from .step_decision import (
    StepDecisionMode,
    StepDecisionModel,
    StepTool,
    TypedToolArguments,
)

_RESULT_TOOL_BASE_NAME = "return_result"


@dataclass(frozen=True)
class NativeToolSet:
    definitions: list[dict[str, Any]]
    result_tool_name: str | None

    @classmethod
    def from_model(cls, model: StepDecisionModel) -> "NativeToolSet":
        names = {tool.name for tool in model.tools}
        result_name = _result_tool_name(names) if model.result is not None else None
        definitions = [_tool_definition(tool) for tool in model.tools]
        if result_name is not None:
            assert model.result is not None
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": result_name,
                        "description": "Return the final result when the task is complete.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "result": _without_titles(
                                    model.result.schema.to_dict()
                                ),
                            },
                            "required": ["result"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return cls(definitions, result_name)

    def validate_calls(
        self, calls: list[ToolCall], model: StepDecisionModel
    ) -> StepDecision:
        if not calls:
            raise ValueError("A native tool call is required.")
        parsed = [_parse_call(call) for call in calls]
        result_calls = [call for call in parsed if call[1] == self.result_tool_name]
        if result_calls:
            if len(parsed) != 1:
                raise ValueError("The result tool cannot be combined with other calls.")
            if model.mode is StepDecisionMode.TOOLS_REQUIRED or model.result is None:
                raise ValueError("The result tool is not allowed.")
            arguments = result_calls[0][2]
            if set(arguments) != {"result"}:
                raise ValueError("The result tool requires exactly the 'result' field.")
            return ResultDecision(
                model.result.validate(LLMOutput.from_json(arguments["result"]))
            )

        if model.mode is StepDecisionMode.RESULT_ONLY:
            raise ValueError("Only the result tool is allowed.")
        tools = {tool.name: tool for tool in model.tools}
        requests: list[ToolCallRequest] = []
        for call_id, name, arguments in parsed:
            tool = tools.get(name)
            if tool is None:
                raise UnknownToolDecisionError(name)
            requests.append(
                ToolCallRequest(
                    id=call_id,
                    name=name,
                    arguments=model.validate_tool_arguments(
                        name, LLMOutput.from_json(arguments)
                    ),
                )
            )
        return ToolCallsDecision(requests)


def _tool_definition(tool: StepTool) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "parameters": (
            _without_titles(tool.arguments.json_schema.to_dict())
            if isinstance(tool.arguments, TypedToolArguments)
            else tool.arguments.json_schema.to_dict()
        ),
    }
    if tool.description:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _without_titles(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: _without_titles(item) for key, item in value.items() if key != "title"
        }
    if isinstance(value, list):
        return [_without_titles(item) for item in value]
    return value


def _result_tool_name(existing: set[str]) -> str:
    name = _RESULT_TOOL_BASE_NAME
    suffix = 2
    while name in existing:
        name = f"{_RESULT_TOOL_BASE_NAME}_{suffix}"
        suffix += 1
    return name


def _parse_call(call: ToolCall) -> tuple[str, str, dict[str, Any]]:
    name = call.function.get("name")
    raw_arguments = call.function.get("arguments")
    if not isinstance(name, str):
        raise ValueError("Native tool call is missing a name.")
    if isinstance(raw_arguments, str):
        arguments = json.loads(raw_arguments)
    else:
        arguments = raw_arguments
    if not isinstance(arguments, dict):
        raise ValueError(f"Native tool call {name!r} arguments must be an object.")
    argument_mapping = cast(dict[object, object], arguments)
    if not all(isinstance(key, str) for key in argument_mapping):
        raise ValueError(f"Native tool call {name!r} arguments must be an object.")
    return call.id, name, cast(dict[str, Any], argument_mapping)


__all__ = ["NativeToolSet"]
