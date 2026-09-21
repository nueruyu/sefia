from copy import deepcopy

from typing_extensions import final, override

from ..._client import LLMClient
from ..._prompt_renderer import PromptRenderer
from ...exceptions import DecisionDecodingError
from .._base import (
    DecisionObserver,
    DecisionRequest,
    DecodedDecision,
    DecisionTransport,
)
from ._decoding import decode_native_tool_calls
from .._messages import (
    application_messages,
    rejection_message,
    response_message,
    text_formatter_for,
)
from ._prompt import native_response_instructions, native_history_messages
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

        messages = application_messages(request, prompt_renderer, ())
        messages.extend(
            native_history_messages(
                request.history, text_formatter_for(prompt_renderer)
            )
        )
        if request.rejected is not None:
            messages.append(rejection_message(request.rejected))
        response = response_message(
            native_response_instructions(request.decision_spec, result_tool)
        )
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
