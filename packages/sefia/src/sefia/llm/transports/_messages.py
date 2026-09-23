from copy import deepcopy
from dataclasses import replace

from .._messages import LLMCompletion, Message
from .._prompt_renderer import PromptRenderer
from .._text import compact_json, json_block, text_block
from ..json_schema import JsonValue
from ..step_decision import StepTool
from ..structured_data import StructuredData
from ._base import (
    DecisionHistoryItem,
    DecisionRequest,
    DecisionToolCalls,
    RejectedDecision,
)


def materialize_application_messages(
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


def response_message(response_instructions: str) -> Message:
    return Message(role="user", content=f"## Response\n\n{response_instructions}")


def rejection_message(rejected: RejectedDecision) -> Message:
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


def append_response(
    messages: list[Message],
    request: DecisionRequest,
    response_instructions: str,
) -> None:
    response = response_message(response_instructions)
    if (
        not request.messages_before
        and not request.messages_after
        and not request.history
        and request.rejected is None
    ):
        prompt_content = messages[0].content
        assert isinstance(prompt_content, str)
        messages[0].content = f"{prompt_content}\n\n{response.content}"
    else:
        messages.append(response)


def _text_history_message(
    history: tuple[DecisionHistoryItem, ...],
) -> Message:
    records: list[StructuredData] = []
    for item in history:
        if isinstance(item, DecisionToolCalls):
            records.extend(
                StructuredData.from_object(
                    {
                        "tool_call": StructuredData.from_object(
                            {
                                "id": StructuredData.from_scalar(call.id),
                                "name": StructuredData.from_scalar(call.name),
                                "arguments": call.arguments,
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
                                "result": item.result,
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
) -> list[Message]:
    messages = materialize_application_messages(request, renderer, tools)
    if request.history:
        messages.append(_text_history_message(request.history))
    if request.rejected is not None:
        messages.append(rejection_message(request.rejected))
    append_response(messages, request, response_instructions)
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
    return compact_json(response)
