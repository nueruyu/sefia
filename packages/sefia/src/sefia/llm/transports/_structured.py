from copy import deepcopy

from typing_extensions import final, override

from .._client import LLMClient
from .._prompt_renderer import PromptRenderer
from ..exceptions import DecisionDecodingError
from ._base import (
    DecisionObserver,
    DecisionRequest,
    DecodedDecision,
    DecisionTransport,
)
from ._decision_instructions import structured_response_instructions
from ._messages import build_decision_messages
from ._text_protocol import text_history_messages


@final
class StructuredDecisionTransport(DecisionTransport):
    @override
    async def request_decision(
        self,
        client: LLMClient,
        prompt_renderer: PromptRenderer,
        request: DecisionRequest,
        observer: DecisionObserver,
        stream: bool,
    ) -> DecodedDecision:
        history_messages = text_history_messages(request.history)
        messages = build_decision_messages(
            request=request,
            renderer=prompt_renderer,
            prompt_tools=request.decision_spec.tools,
            history_messages=history_messages,
            response_instructions=structured_response_instructions(
                request.decision_spec
            ),
        )
        await observer.before_request(tuple(deepcopy(messages)))

        completion = await client.complete(
            messages=messages,
            tools=None,
            decision_spec=request.decision_spec,
            stream_callback=observer.response_text if stream else None,
            output_callback=observer.output if stream else None,
            reasoning_callback=observer.reasoning_text if stream else None,
        )
        data = completion.structured_output
        if data is None:
            raise DecisionDecodingError(
                completion, "LLM client did not return structured output."
            )
        return DecodedDecision(decision_data=data, completion=completion)
