from collections.abc import Callable
from copy import deepcopy

from typing_extensions import final, override

from ..._client import LLMClient
from ..._prompt_renderer import PromptRenderer
from ...exceptions import DecisionDecodingError
from ...structured_data import StructuredData
from .._base import (
    DecisionObserver,
    DecisionRequest,
    DecodedDecision,
    DecisionTransport,
)
from ._decoding import decode_native_tool_calls
from ._messages import build_native_messages
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
        *,
        dump: Callable[[object], StructuredData],
    ) -> DecodedDecision:
        result_tool = create_result_tool(request.decision_spec)
        tools = [*request.decision_spec.tools]
        if result_tool is not None:
            tools.append(result_tool)

        messages = build_native_messages(
            request=request,
            renderer=prompt_renderer,
            result_tool=result_tool,
            dump=dump,
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
