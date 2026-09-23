import json
from copy import deepcopy
from dataclasses import replace

from .._markdown import text_block
from .._messages import LLMCompletion, Message
from .._prompt_renderer import PromptRenderer
from ..json_schema import JsonValue
from ..step_decision import StepTool
from ._base import DecisionRequest, RejectedDecision


def _build_application_messages(
    request: DecisionRequest,
    renderer: PromptRenderer,
    tools: tuple[StepTool, ...],
) -> list[Message]:
    prompt = replace(
        request.inference_prompt,
        arguments=deepcopy(request.inference_prompt.arguments),
        tools=tools,
    )
    return [
        *deepcopy(request.messages_before),
        Message(role="user", content=renderer.render(prompt)),
        *deepcopy(request.messages_after),
    ]


def _response_message(response_instructions: str) -> Message:
    return Message(role="user", content=f"## Response\n\n{response_instructions}")


def _rejection_message(rejected: RejectedDecision) -> Message:
    content = _rejected_completion_content(rejected.completion)
    previous = (
        "The previous response was empty."
        if not content
        else "Previous response:\n" + text_block(content)
    )
    return Message(
        role="user",
        content=(
            "## Correct the previous response\n\n"
            f"{previous}\n\nReason: {rejected.reason}\n\n"
            "Return a corrected response matching the Response section."
        ),
    )


def build_decision_messages(
    *,
    request: DecisionRequest,
    renderer: PromptRenderer,
    prompt_tools: tuple[StepTool, ...],
    history_messages: list[Message],
    response_instructions: str,
) -> list[Message]:
    messages = _build_application_messages(request, renderer, prompt_tools)
    messages.extend(history_messages)
    if request.rejected is not None:
        messages.append(_rejection_message(request.rejected))
    response = _response_message(response_instructions)
    if (
        not request.messages_before
        and not request.messages_after
        and not history_messages
        and request.rejected is None
    ):
        prompt_content = messages[0].content
        assert isinstance(prompt_content, str)
        messages[0].content = f"{prompt_content}\n\n{response.content}"
    else:
        messages.append(response)
    return messages


def _rejected_completion_content(completion: LLMCompletion) -> str | None:
    if (
        not completion.tool_calls
        and completion.structured_output is None
        and completion.content is not None
    ):
        return completion.content
    if (
        completion.content is None
        and not completion.tool_calls
        and completion.structured_output is None
    ):
        return None

    response: dict[str, JsonValue] = {}
    if completion.content is not None:
        response["content"] = completion.content
    if completion.tool_calls:
        response["tool_calls"] = [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments.to_json_value(),
            }
            for call in completion.tool_calls
        ]
    if completion.structured_output is not None:
        response["structured_output"] = completion.structured_output.to_json_value()
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))
