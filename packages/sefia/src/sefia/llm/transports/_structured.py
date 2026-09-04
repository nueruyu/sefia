from typing_extensions import final, override

from .._client import LLMClient
from .._messages import Message
from .._prompt_renderer import PromptRenderer
from ._base import (
    DecisionDecodingError,
    DecisionObserver,
    DecisionRequest,
    DecisionResponse,
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
    ) -> DecisionResponse:
        prompt = prompt_renderer.render(
            request.to_prompt(
                json_response_instructions(request.spec),
                tools=request.spec.tools,
                history=request.history,
            )
        )
        await observer.before_request(prompt)

        response = await client.complete(
            messages=[Message(role="user", content=prompt)],
            tools=None,
            decision_model=request.spec,
            stream_callback=observer.response_text if stream else None,
            output_callback=observer.output if stream else None,
            reasoning_callback=observer.reasoning_text if stream else None,
        )
        output = response.structured_output
        if output is None:
            raise DecisionDecodingError(
                response, "LLM client did not return structured output."
            )
        return DecisionResponse(output=output, raw=response)
