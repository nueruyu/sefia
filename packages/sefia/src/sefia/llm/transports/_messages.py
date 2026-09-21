from copy import deepcopy

from ...inference import HistoryItem, ToolCallsDecision
from .._markdown_prompt_renderer import MarkdownPromptRenderer
from .._messages import Message
from .._prompt_renderer import InferencePrompt, PromptRenderer, RejectedDecision
from .._text import TextFormatter, text_block
from ..step_decision import StepTool
from ._base import DecisionRequest


def text_formatter_for(renderer: PromptRenderer) -> TextFormatter:
    if isinstance(renderer, MarkdownPromptRenderer):
        return renderer.text_formatter
    return TextFormatter()


def application_messages(
    request: DecisionRequest,
    renderer: PromptRenderer,
    tools: tuple[StepTool, ...],
) -> list[Message]:
    layout = request.message_layout
    prompt = InferencePrompt(
        function=request.function,
        arguments=layout.arguments,
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


def _text_history_message(
    history: tuple[HistoryItem, ...], formatter: TextFormatter
) -> Message:
    records: list[object] = []
    for item in history:
        if isinstance(item, ToolCallsDecision):
            records.extend(
                {
                    "tool_call": {
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                }
                for call in item.calls
            )
        else:
            records.append(
                {
                    "tool_result": {
                        "id": item.tool_call_id,
                        "result": item.result,
                    }
                }
            )
    return Message(
        role="user",
        content="## Previous tool interactions\n\n" + formatter.json_block(records),
    )


def text_protocol_messages(
    request: DecisionRequest,
    renderer: PromptRenderer,
    tools: tuple[StepTool, ...],
    response_instructions: str,
) -> list[Message]:
    messages = application_messages(request, renderer, tools)
    if request.history:
        messages.append(
            _text_history_message(request.history, text_formatter_for(renderer))
        )
    if request.rejected is not None:
        messages.append(rejection_message(request.rejected))
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
    return messages
