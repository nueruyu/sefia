import json
from collections.abc import Callable, Sequence

from ..exceptions import InvalidInferenceResponseError
from ..inference import FunctionInfo, HistoryItem, ToolCallsDecision
from ._messages import Message
from ._prompt_renderer import PromptRenderer
from .step_decision import StepDecisionSpec

JsonDefault = Callable[[object], object]


def build_messages(
    function_info: FunctionInfo,
    history: Sequence[HistoryItem],
    decision_spec: StepDecisionSpec,
    prompt_renderer: PromptRenderer,
    json_default: JsonDefault | None,
) -> list[Message]:
    messages = [
        Message(
            role="system",
            content=prompt_renderer.render_instructions(
                function_info,
                decision_spec,
            ),
        ),
        Message(
            role="user",
            content=prompt_renderer.render_invocation(function_info),
        ),
    ]

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
                        separators=(",", ":"),
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
                        separators=(",", ":"),
                    ),
                )
            )

    return messages


def build_response_feedback_messages(
    error: InvalidInferenceResponseError,
    prompt_renderer: PromptRenderer,
) -> list[Message]:
    messages: list[Message] = []
    if error.raw_content:
        messages.append(Message(role="assistant", content=error.raw_content))
    messages.append(
        Message(
            role="user",
            content=prompt_renderer.render_response_feedback(error),
        )
    )
    return messages
