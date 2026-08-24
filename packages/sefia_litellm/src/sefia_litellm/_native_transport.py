from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

from typing_extensions import final, override

from sefia.exceptions import InvalidInferenceResponseError, UnknownToolDecisionError
from sefia.inference import (
    HistoryItem,
    ResultDecision,
    ToolCallRequest,
    ToolCallsDecision,
)
from sefia.llm import LLMOutput, LLMResponse, Message
from sefia.llm._messages import ToolCall
from sefia.llm._tool_call_ids import ToolCallIdRegistry
from sefia.llm.json_schema import without_titles
from sefia.llm.step_decision import (
    StepDecisionMode,
    StepDecisionModel,
    StepDecisionSpec,
    TypedToolArguments,
)
from sefia.llm.transports import (
    JsonDefault,
    ResultTransport,
    ToolCallTransport,
    ToolDefinition,
)

_RESULT_TOOL_BASE_NAME = "return_result"


def _result_tool_name(model: StepDecisionModel) -> str:
    existing = {tool.name for tool in model.tools}
    name = _RESULT_TOOL_BASE_NAME
    suffix = 2
    while name in existing:
        name = f"{_RESULT_TOOL_BASE_NAME}_{suffix}"
        suffix += 1
    return name


def _arguments(call: ToolCall) -> dict[str, Any]:
    value = (
        json.loads(call.arguments)
        if isinstance(call.arguments, str)
        else call.arguments
    )
    if not isinstance(value, dict):
        raise ValueError(f"Native tool call {call.name!r} arguments must be an object.")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"Native tool call {call.name!r} arguments must be an object.")
    return cast(dict[str, Any], mapping)


def _repair(error: InvalidInferenceResponseError) -> list[Message]:
    return [
        Message(
            role="user",
            content=(
                "Your previous response could not be used.\n"
                f"Error: {error.detail}\n"
                "Respond by calling exactly the appropriate available tool. "
                "Do not answer with text."
            ),
        )
    ]


@final
class NativeToolCallTransport(ToolCallTransport):
    @property
    @override
    def supports_arg_streaming(self) -> bool:
        return False

    @override
    def definitions(self, model: StepDecisionModel) -> list[ToolDefinition] | None:
        definitions: list[ToolDefinition] = []
        for tool in model.tools:
            schema = tool.arguments.json_schema.to_dict()
            definitions.append(
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=cast(
                        dict[str, Any],
                        without_titles(schema)
                        if isinstance(tool.arguments, TypedToolArguments)
                        else schema,
                    ),
                )
            )
        return definitions or None

    @override
    def decision_model(self, model: StepDecisionModel) -> StepDecisionModel | None:
        del model
        return None

    @override
    def prompt(self, spec: StepDecisionSpec, model: StepDecisionModel) -> str:
        del spec, model
        return (
            "\n\nCall available tools when needed. Do not describe a tool call in text."
        )

    @override
    def render_history(
        self, history: Sequence[HistoryItem], json_default: JsonDefault | None
    ) -> list[Message]:
        messages: list[Message] = []
        for item in history:
            if isinstance(item, ToolCallsDecision):
                messages.append(
                    Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id=call.id,
                                name=call.name,
                                arguments=json.dumps(
                                    call.arguments,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            )
                            for call in item.calls
                        ],
                    )
                )
            else:
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=item.tool_call_id,
                        content=json.dumps(
                            item.result,
                            default=json_default,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    )
                )
        return messages

    @override
    def decode(
        self,
        response: LLMResponse,
        model: StepDecisionModel,
        tool_call_ids: ToolCallIdRegistry,
    ) -> ToolCallsDecision | None:
        del tool_call_ids
        if not response.tool_calls:
            return None
        result_name = _result_tool_name(model) if model.result is not None else None
        if any(call.name == result_name for call in response.tool_calls):
            return None
        tools = {tool.name for tool in model.tools}
        requests: list[ToolCallRequest] = []
        for call in response.tool_calls:
            if call.name not in tools:
                raise UnknownToolDecisionError(call.name)
            arguments = _arguments(call)
            requests.append(
                ToolCallRequest(
                    call.id,
                    call.name,
                    model.validate_tool_arguments(
                        call.name, LLMOutput.from_json(arguments)
                    ),
                )
            )
        return ToolCallsDecision(requests)

    @override
    def repair_messages(self, error: InvalidInferenceResponseError) -> list[Message]:
        return _repair(error)


@final
class NativeResultTransport(ResultTransport):
    @override
    def definitions(self, model: StepDecisionModel) -> list[ToolDefinition] | None:
        if model.result is None:
            return None
        return [
            ToolDefinition(
                name=_result_tool_name(model),
                description="Return the final result when the task is complete.",
                parameters={
                    "type": "object",
                    "properties": {
                        "result": without_titles(model.result.schema.to_dict())
                    },
                    "required": ["result"],
                    "additionalProperties": False,
                },
            )
        ]

    @override
    def decision_model(self, model: StepDecisionModel) -> StepDecisionModel | None:
        del model
        return None

    @override
    def prompt(self, spec: StepDecisionSpec, model: StepDecisionModel) -> str:
        if model.result is None:
            return "\n\nDo not return a final answer."
        name = _result_tool_name(model)
        if spec.mode is StepDecisionMode.RESULT_ONLY:
            return f"\n\nCall `{name}` with the final result. Do not answer with text."
        return f"\n\nWhen the task is complete, call `{name}` with the final result."

    @override
    def decode(
        self, response: LLMResponse, model: StepDecisionModel
    ) -> ResultDecision | None:
        if model.result is None:
            return None
        name = _result_tool_name(model)
        result_calls = [call for call in response.tool_calls if call.name == name]
        if not result_calls:
            return None
        if len(response.tool_calls) != 1:
            raise ValueError("The result tool cannot be combined with other calls.")
        arguments = _arguments(result_calls[0])
        if set(arguments) != {"result"}:
            raise ValueError("The result tool requires exactly the 'result' field.")
        return ResultDecision(
            model.result.validate(LLMOutput.from_json(arguments["result"]))
        )

    @override
    def repair_messages(self, error: InvalidInferenceResponseError) -> list[Message]:
        return _repair(error)


__all__ = ["NativeResultTransport", "NativeToolCallTransport"]
