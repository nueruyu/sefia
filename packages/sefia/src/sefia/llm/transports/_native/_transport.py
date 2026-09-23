import json
from copy import deepcopy
from dataclasses import replace

from typing_extensions import final, override

from ..._client import LLMClient
from ..._markdown import text_block
from ..._messages import LLMCompletion, Message
from ..._prompt_renderer import PromptRenderer
from ...exceptions import DecisionDecodingError
from ...json_schema import JsonValue
from ...step_decision import StepTool
from .._base import (
    DecisionObserver,
    DecisionRequest,
    DecodedDecision,
    DecisionTransport,
    RejectedDecision,
)
from ._decoding import decode_native_tool_calls
from ._prompt import native_history_messages, native_response_instructions
from ._result_tool import create_result_tool


@final
class NativeDecisionTransport(DecisionTransport):
    """Represents decisions with native tool calls and tool-result messages."""

    @override
    async def request_decision(
        self,
        client: LLMClient,
        prompt_renderer: PromptRenderer,
        request: DecisionRequest,
        observer: DecisionObserver,
        stream: bool,
    ) -> DecodedDecision:
        result_tool = create_result_tool(request.decision_spec)
        tools = [*request.decision_spec.tools]
        if result_tool is not None:
            tools.append(result_tool)

        messages = _build_messages(
            request=request,
            renderer=prompt_renderer,
            result_tool=result_tool,
        )
        await observer.before_request(tuple(deepcopy(messages)))

        completion = await client.complete(
            messages=messages,
            tools=tools,
            decision_spec=None,
            stream_callback=observer.response_text if stream else None,
            output_callback=observer.output if stream else None,
            reasoning_callback=observer.reasoning_text if stream else None,
        )
        try:
            data = decode_native_tool_calls(completion.tool_calls, result_tool)
        except ValueError as error:
            raise DecisionDecodingError(completion, str(error)) from error
        return DecodedDecision(decision_data=data, completion=completion)


def _build_messages(
    *,
    request: DecisionRequest,
    renderer: PromptRenderer,
    result_tool: StepTool | None,
) -> list[Message]:
    messages = _build_application_messages(request, renderer, ())
    messages.extend(native_history_messages(request.history))
    if request.rejected is not None:
        messages.append(_rejection_message(request.rejected))
    _append_response(
        messages,
        request,
        native_response_instructions(request.decision_spec, result_tool),
    )
    return messages


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


def _append_response(
    messages: list[Message],
    request: DecisionRequest,
    response_instructions: str,
) -> None:
    response = Message(role="user", content=f"## Response\n\n{response_instructions}")
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
