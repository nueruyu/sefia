from typing import cast

from ..._message_plan import TaskPrompt
from .._messages import Message
from .._prompt_renderer import PromptRenderer
from ..step_decision import StepTool
from ._base import DecisionRequest


def materialize_plan(
    request: DecisionRequest,
    renderer: PromptRenderer,
    response_instructions: str,
    tools: tuple[StepTool, ...],
) -> list[Message]:
    messages: list[Message] = []
    for planned_part in request.message_plan.parts:
        part = cast(object, planned_part)
        if isinstance(part, Message):
            messages.append(part)
        elif isinstance(part, TaskPrompt):
            prompt = request.to_prompt(
                response_instructions, arguments=part.arguments, tools=tools
            )
            messages.append(Message(role="user", content=renderer.render(prompt)))
        else:
            raise TypeError(f"Unknown MessagePlan part type: {type(part).__name__}")
    return messages


def append_text_feedback(
    messages: list[Message], request: DecisionRequest, renderer: PromptRenderer
) -> None:
    if request.history:
        messages.append(
            Message(role="user", content=renderer.render_history(request.history))
        )
    if request.rejected is not None:
        messages.append(
            Message(role="user", content=renderer.render_rejection(request.rejected))
        )
