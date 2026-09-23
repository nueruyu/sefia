from collections.abc import Callable
from copy import deepcopy

from ...inference import HistoryItem, ToolCallsDecision
from .._messages import Message
from .._prompt_renderer import InferencePrompt, PromptRenderer
from .._text import json_block, text_block
from ..step_decision import StepTool
from ..structured_data import StructuredData
from ._base import DecisionRequest, RejectedDecision


def materialize_application_messages(
    request: DecisionRequest,
    renderer: PromptRenderer,
    tools: tuple[StepTool, ...],
    dump: Callable[[object], StructuredData],
) -> list[Message]:
    layout = request.message_layout
    arguments = StructuredData.from_object(
        {name: dump(value) for name, value in layout.arguments.items()}
    )
    prompt = InferencePrompt(
        function=request.function,
        arguments=arguments,
        tools=tools,
    )
    return [
        *deepcopy(layout.before),
        Message(role="user", content=renderer.render(prompt)),
        *deepcopy(layout.after),
    ]


def response_message(response_instructions: str) -> Message:
    return Message(role="user", content=f"## Response\n\n{response_instructions}")


def rejection_message(rejected: RejectedDecision) -> Message:
    previous = (
        "The previous response was empty."
        if not rejected.content
        else "Previous response:\n" + text_block(rejected.content)
    )
    return Message(
        role="user",
        content=(
            "## Correct the previous response\n\n"
            f"{previous}\n\nReason: {rejected.reason}\n\n"
            "Return a corrected response matching the Response section."
        ),
    )


def append_response(
    messages: list[Message],
    request: DecisionRequest,
    response_instructions: str,
) -> None:
    response = response_message(response_instructions)
    if (
        not request.message_layout.before
        and not request.message_layout.after
        and not request.history
        and request.rejected is None
    ):
        prompt_content = messages[0].content
        assert isinstance(prompt_content, str)
        messages[0].content = f"{prompt_content}\n\n{response.content}"
    else:
        messages.append(response)


def _text_history_message(
    history: tuple[HistoryItem, ...],
    dump: Callable[[object], StructuredData],
) -> Message:
    records: list[StructuredData] = []
    for item in history:
        if isinstance(item, ToolCallsDecision):
            records.extend(
                StructuredData.from_object(
                    {
                        "tool_call": StructuredData.from_object(
                            {
                                "id": StructuredData.from_scalar(call.id),
                                "name": StructuredData.from_scalar(call.name),
                                "arguments": dump(call.arguments),
                            }
                        )
                    }
                )
                for call in item.calls
            )
        else:
            records.append(
                StructuredData.from_object(
                    {
                        "tool_result": StructuredData.from_object(
                            {
                                "id": StructuredData.from_scalar(item.tool_call_id),
                                "result": dump(item.result),
                            }
                        )
                    }
                )
            )
    data = StructuredData.from_array(records)
    return Message(
        role="user",
        content="## Previous tool interactions\n\n" + json_block(data.to_json_value()),
    )


def build_text_messages(
    *,
    request: DecisionRequest,
    renderer: PromptRenderer,
    tools: tuple[StepTool, ...],
    response_instructions: str,
    dump: Callable[[object], StructuredData],
) -> list[Message]:
    messages = materialize_application_messages(request, renderer, tools, dump)
    if request.history:
        messages.append(_text_history_message(request.history, dump))
    if request.rejected is not None:
        messages.append(rejection_message(request.rejected))
    append_response(messages, request, response_instructions)
    return messages
