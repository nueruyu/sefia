import json
from collections.abc import Sequence
from typing import Any, Callable

from ..inference import FunctionInfo, HistoryItem, ToolCallsDecision
from ._messages import Message
from ._prompt_formatter import PromptFormatter

JsonDefault = Callable[[Any], Any]


def build_initial_messages(
    function_info: FunctionInfo,
    system_prompt_addition: str,
    prompt_formatter: PromptFormatter,
) -> list[Message]:
    prompt_arguments = function_info.prompt_arguments
    user_prompt = (
        prompt_formatter.format_arguments(prompt_arguments)
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


def render_envelope_history(
    history: Sequence[HistoryItem], json_default: JsonDefault | None
) -> list[Message]:
    return _render_json_history(history, json_default, correlate_by_id=True)


def render_prompt_json_history(
    history: Sequence[HistoryItem], json_default: JsonDefault | None
) -> list[Message]:
    return _render_json_history(history, json_default, correlate_by_id=False)


def _render_json_history(
    history: Sequence[HistoryItem],
    json_default: JsonDefault | None,
    *,
    correlate_by_id: bool,
) -> list[Message]:
    messages: list[Message] = []
    calls_by_id: dict[str, dict[str, Any]] = {}
    for item in history:
        if isinstance(item, ToolCallsDecision):
            calls = [
                {"name": call.name, "arguments": call.arguments} for call in item.calls
            ]
            if not correlate_by_id:
                calls_by_id.update(
                    (call.id, rendered)
                    for call, rendered in zip(item.calls, calls, strict=True)
                )
            rendered_calls = (
                [
                    {"id": call.id, **rendered}
                    for call, rendered in zip(item.calls, calls, strict=True)
                ]
                if correlate_by_id
                else calls
            )
            messages.append(
                Message(
                    role="assistant",
                    content=json.dumps(
                        {"decision": "tool_calls", "tool_calls": rendered_calls},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )
            continue

        call = calls_by_id.get(item.tool_call_id)
        tool_result = (
            {"tool_call_id": item.tool_call_id, "result": item.result}
            if correlate_by_id or call is None
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


__all__ = [
    "build_initial_messages",
    "render_envelope_history",
    "render_prompt_json_history",
]
