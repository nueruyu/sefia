from copy import deepcopy
from typing import cast

from .._message_plan import TaskPrompt
from .._messages import Message
from .._prompt_renderer import PromptRenderer
from ..step_decision import StepTool
from ._base import DecisionRequest


def materialize_plan(
    request: DecisionRequest,
    renderer: PromptRenderer,
    tools: tuple[StepTool, ...],
) -> list[Message]:
    messages: list[Message] = []
    for planned_part in request.message_plan.parts:
        part = cast(object, planned_part)
        if isinstance(part, Message):
            messages.append(deepcopy(part))
        elif isinstance(part, TaskPrompt):
            prompt = request.to_prompt(arguments=part.arguments, tools=tools)
            messages.append(Message(role="user", content=renderer.render(prompt)))
        else:
            raise TypeError(f"Unknown MessagePlan part type: {type(part).__name__}")
    return messages


def append_decision_instructions(
    messages: list[Message],
    request: DecisionRequest,
    renderer: PromptRenderer,
    instructions: str,
) -> None:
    control = renderer.render_decision_instructions(instructions)
    if (
        len(request.message_plan.parts) == 1
        and not request.history
        and request.rejected is None
    ):
        task_content = messages[0].content
        assert isinstance(task_content, str)
        messages[0].content = f"{task_content}\n\n{control}"
    else:
        messages.append(Message(role="user", content=control))


def snapshot_messages(messages: list[Message]) -> tuple[Message, ...]:
    return tuple(deepcopy(messages))


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
