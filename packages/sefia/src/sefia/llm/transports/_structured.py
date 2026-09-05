from typing_extensions import final, override

from .._client import LLMClient
from .._messages import Message
from .._prompt_renderer import PromptRenderer
from ..exceptions import DecisionDecodingError
from ._base import (
    DecisionObserver,
    DecisionRequest,
    DecodedDecision,
    DecisionTransport,
)
from ._json_decision import json_response_instructions


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
        prompt = prompt_renderer.render(
            request.to_prompt(
                json_response_instructions(request.spec),
                tools=request.spec.tools,
                history=request.history,
            )
        )
        await observer.before_request(prompt)

        completion = await client.complete(
            messages=[Message(role="user", content=prompt)],
            tools=None,
            decision_spec=request.spec,
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
