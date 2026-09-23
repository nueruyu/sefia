from ..._messages import Message
from ..._prompt_renderer import PromptRenderer
from ...step_decision import StepTool
from .._base import DecisionRequest
from .._messages import (
    append_response,
    materialize_application_messages,
    rejection_message,
)
from ._prompt import native_history_messages, native_response_instructions


def build_native_messages(
    *,
    request: DecisionRequest,
    renderer: PromptRenderer,
    result_tool: StepTool | None,
) -> list[Message]:
    messages = materialize_application_messages(request, renderer, ())
    messages.extend(native_history_messages(request.history))
    if request.rejected is not None:
        messages.append(rejection_message(request.rejected))
    append_response(
        messages,
        request,
        native_response_instructions(request.decision_spec, result_tool),
    )
    return messages


__all__ = ["build_native_messages"]
