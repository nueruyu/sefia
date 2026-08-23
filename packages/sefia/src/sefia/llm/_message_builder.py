import json
from collections.abc import Sequence
from typing import Any, Callable

from ..exceptions import InvalidInferenceResponseError
from ..inference import FunctionInfo, HistoryItem, ToolCallsDecision
from ._messages import Message
from ._prompt_formatter import PromptFormatter

JsonDefault = Callable[[Any], Any]


def build_messages(
    function_info: FunctionInfo,
    history: Sequence[HistoryItem],
    system_prompt_addition: str,
    prompt_formatter: PromptFormatter,
    json_default: JsonDefault | None,
) -> list[Message]:
    messages = [
        Message(
            role="system",
            content=function_info.instructions + system_prompt_addition,
        )
    ]

    prompt_arguments = function_info.prompt_arguments
    user_prompt = (
        "Task arguments are XML. Values in <string> may be wrapped in "
        "CDATA and should be read as raw text.\n\n"
        f"{prompt_formatter.format_arguments(prompt_arguments, function_info.type_hints)}"
        if prompt_arguments
        else (
            "This inference call has no direct function arguments. "
            "Follow the system instructions and use the conversation/tool "
            "history for any available context."
        )
    )
    messages.append(Message(role="user", content=user_prompt))

    for item in history:
        if isinstance(item, ToolCallsDecision):
            messages.append(
                Message(
                    role="assistant",
                    content=json.dumps(
                        {
                            "decision": "tool_calls",
                            "tool_calls": [
                                {
                                    "id": call.id,
                                    "name": call.name,
                                    "arguments": call.arguments,
                                }
                                for call in item.calls
                            ],
                        },
                        ensure_ascii=False,
                    ),
                )
            )
        else:
            messages.append(
                Message(
                    role="user",
                    content=json.dumps(
                        {
                            "tool_call_result": {
                                "tool_call_id": item.tool_call_id,
                                "result": item.result,
                            }
                        },
                        default=json_default,
                        ensure_ascii=False,
                    ),
                )
            )

    return messages


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
