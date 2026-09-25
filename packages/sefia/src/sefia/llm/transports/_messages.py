import json

from .._json import JsonSnapshot
from .._markdown import json_block, text_block
from .._messages import LLMCompletion, Message
from .._prompt_renderer import InferencePrompt, PromptRenderer
from ..step_decision import StepTool
from ._base import (
    DecisionHistoryItem,
    DecisionRequest,
    DecisionToolCalls,
    RejectedDecision,
)


def _build_application_messages(
    request: DecisionRequest,
    renderer: PromptRenderer,
    tools: tuple[StepTool, ...],
) -> list[Message]:
    prompt = InferencePrompt(
        function=request.function,
        arguments=request.arguments,
        tools=tools,
    )
    return [
        *request.messages_before,
        Message(role="user", content=renderer.render(prompt)),
        *request.messages_after,
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


def _text_history_messages(
    history: tuple[DecisionHistoryItem, ...],
) -> list[Message]:
    if not history:
        return []

    records: list[JsonSnapshot] = []
    for item in history:
        if isinstance(item, DecisionToolCalls):
            records.extend(
                JsonSnapshot.from_object(
                    {
                        "tool_call": JsonSnapshot.from_object(
                            {
                                "id": JsonSnapshot.from_scalar(call.id),
                                "name": JsonSnapshot.from_scalar(call.name),
                                "arguments": call.arguments,
                            }
                        )
                    }
                )
                for call in item.calls
            )
        else:
            records.append(
                JsonSnapshot.from_object(
                    {
                        "tool_result": JsonSnapshot.from_object(
                            {
                                "id": JsonSnapshot.from_scalar(item.tool_call_id),
                                "result": item.result,
                            }
                        )
                    }
                )
            )

    data = JsonSnapshot.from_array(records)
    return [
        Message(
            role="user",
            content=(
                "## Previous tool interactions\n\n"
                + json_block(data.to_json_compatible())
            ),
        )
    ]


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
        response_content = response.content
        assert isinstance(prompt_content, str)
        assert isinstance(response_content, str)
        messages[0] = Message(
            role="user",
            content=f"{prompt_content}\n\n{response_content}",
        )
    else:
        messages.append(response)
    return messages


def build_text_decision_messages(
    *,
    request: DecisionRequest,
    renderer: PromptRenderer,
    response_instructions: str,
) -> list[Message]:
    return build_decision_messages(
        request=request,
        renderer=renderer,
        prompt_tools=request.decision_spec.tools,
        history_messages=_text_history_messages(request.history),
        response_instructions=response_instructions,
    )


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

    response: dict[str, JsonSnapshot] = {}
    if completion.content is not None:
        response["content"] = JsonSnapshot.from_scalar(completion.content)
    if completion.tool_calls:
        response["tool_calls"] = JsonSnapshot.from_array(
            JsonSnapshot.from_object(
                {
                    "id": JsonSnapshot.from_scalar(call.id),
                    "name": JsonSnapshot.from_scalar(call.name),
                    "arguments": call.arguments,
                }
            )
            for call in completion.tool_calls
        )
    if completion.structured_output is not None:
        response["structured_output"] = completion.structured_output
    return json.dumps(
        JsonSnapshot.from_object(response).to_json_compatible(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
