import json
from copy import deepcopy

from ..._messages import Message, ToolCall
from ...step_decision import DecisionSpec, StepDecisionMode, StepTool
from .._base import DecisionHistoryItem, DecisionToolCalls


def native_response_instructions(
    spec: DecisionSpec,
    result_tool: StepTool | None,
) -> str:
    if spec.mode is StepDecisionMode.TOOLS_REQUIRED:
        action = "Call one or more available tools."
    elif spec.mode is StepDecisionMode.RESULT_ONLY:
        assert result_tool is not None
        action = f"Call `{result_tool.name}` with the final result."
    else:
        assert result_tool is not None
        action = (
            "Call available tools when needed. When the task is complete, "
            f"call `{result_tool.name}` with the final result."
        )
    return f"{action}\nDo not describe a tool call or answer with text."


def native_history_messages(
    history: tuple[DecisionHistoryItem, ...],
) -> list[Message]:
    messages: list[Message] = []
    for item in history:
        if isinstance(item, DecisionToolCalls):
            messages.append(
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id=call.id,
                            name=call.name,
                            arguments=deepcopy(call.arguments),
                        )
                        for call in item.calls
                    ],
                )
            )
        else:
            messages.append(
                Message(
                    role="tool",
                    content=json.dumps(
                        item.result.to_json_value(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    tool_call_id=item.tool_call_id,
                )
            )
    return messages
