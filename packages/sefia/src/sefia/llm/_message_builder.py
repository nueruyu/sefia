import json
from collections.abc import Sequence
from typing import Any, Callable

from ..exceptions import InvalidInferenceResponseError
from ..inference import FunctionInfo, HistoryItem, ToolCallsDecision
from ._messages import Message, ToolCall
from ._prompt_formatter import PromptFormatter

JsonDefault = Callable[[Any], Any]


def build_messages(
    function_info: FunctionInfo,
    history: Sequence[HistoryItem],
    system_prompt_addition: str,
    prompt_formatter: PromptFormatter,
    json_default: JsonDefault | None,
    include_tool_call_ids: bool = True,
) -> list[Message]:
    messages = _initial_messages(
        function_info, system_prompt_addition, prompt_formatter
    )

    calls_by_id: dict[str, dict[str, Any]] = {}
    for item in history:
        if isinstance(item, ToolCallsDecision):
            calls = [
                {"name": call.name, "arguments": call.arguments} for call in item.calls
            ]
            for call, rendered in zip(item.calls, calls, strict=True):
                calls_by_id[call.id] = rendered
            messages.append(
                Message(
                    role="assistant",
                    content=json.dumps(
                        {
                            "decision": "tool_calls",
                            "tool_calls": (
                                [
                                    {"id": call.id, **rendered}
                                    for call, rendered in zip(
                                        item.calls, calls, strict=True
                                    )
                                ]
                                if include_tool_call_ids
                                else calls
                            ),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )
        else:
            call = calls_by_id.get(item.tool_call_id)
            tool_result = (
                {"tool_call_id": item.tool_call_id, "result": item.result}
                if include_tool_call_ids or call is None
                else {**call, "result": item.result}
            )
            messages.append(
                Message(
                    role="user",
                    content=json.dumps(
                        {"tool_call_result": tool_result},
                        default=json_default,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )

    return messages


def build_native_messages(
    function_info: FunctionInfo,
    history: Sequence[HistoryItem],
    system_prompt_addition: str,
    prompt_formatter: PromptFormatter,
    json_default: JsonDefault | None,
) -> list[Message]:
    messages = _initial_messages(
        function_info, system_prompt_addition, prompt_formatter
    )
    for item in history:
        if isinstance(item, ToolCallsDecision):
            messages.append(
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id=call.id,
                            function={
                                "name": call.name,
                                "arguments": json.dumps(
                                    call.arguments,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
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


def _initial_messages(
    function_info: FunctionInfo,
    system_prompt_addition: str,
    prompt_formatter: PromptFormatter,
) -> list[Message]:
    prompt_arguments = function_info.prompt_arguments
    user_prompt = (
        prompt_formatter.format_arguments(prompt_arguments, function_info.type_hints)
        if prompt_arguments
        else (
            "This inference call has no direct function arguments. "
            "Follow the system instructions and use the conversation/tool "
            "history for any available context."
        )
    )
    return [
        Message(
            role="system",
            content=function_info.instructions + system_prompt_addition,
        ),
        Message(role="user", content=user_prompt),
    ]


def build_repair_messages(error: InvalidInferenceResponseError) -> list[Message]:
    messages: list[Message] = []
    if error.raw_content:
        messages.append(Message(role="assistant", content=error.raw_content))
        content_note = ""
    else:
        content_note = "Your previous response was empty.\n"

    feedback = (
        "Your previous response was invalid and could not be used as the "
        "required decision JSON.\n"
        f"Error: {error.detail}\n"
        f"{content_note}"
        "Respond again with exactly one valid raw JSON object matching the "
        "step-decision schema in the system instructions. Do not include prose, "
        "markdown, or code fences."
    )
    messages.append(Message(role="user", content=feedback))
    return messages


def build_native_repair_messages(error: InvalidInferenceResponseError) -> list[Message]:
    feedback = (
        "Your previous response could not be used.\n"
        f"Error: {error.detail}\n"
        "Respond by calling exactly the appropriate available tool. Do not answer "
        "with text."
    )
    return [Message(role="user", content=feedback)]
