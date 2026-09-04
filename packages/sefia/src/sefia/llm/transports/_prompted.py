from typing_extensions import final, override

from .._client import LLMClient
from .._messages import Message
from .._prompted_response import PromptedJsonStreamExtractor, extract_prompted_json
from .._prompt_renderer import PromptRenderer
from ..llm_output import LLMOutput
from ..streaming import JsonOutputStreamDecoder
from ._base import (
    DecisionDecodingError,
    DecisionObserver,
    DecisionRequest,
    DecisionResponse,
    DecisionTransport,
)
from ._json_decision import json_response_instructions


@final
class PromptedDecisionTransport(DecisionTransport):
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
        stream_decoder = JsonOutputStreamDecoder() if stream else None
        extractor = PromptedJsonStreamExtractor() if stream else None

        async def on_text(text: str) -> None:
            assert stream_decoder is not None and extractor is not None
            await observer.response_text(text)
            json_text = extractor.feed(text)
            if json_text:
                for event in stream_decoder.feed(json_text):
                    await observer.output(event)

        response = await client.complete(
            messages=[Message(role="user", content=prompt)],
            tools=None,
            decision_model=None,
            stream_callback=on_text if stream else None,
            output_callback=None,
            reasoning_callback=observer.reasoning_text if stream else None,
        )
        if response.content is None:
            raise DecisionDecodingError(
                response, "LLM did not provide response content."
            )
        try:
            output = LLMOutput.parse_json(extract_prompted_json(response.content))
        except ValueError as error:
            raise DecisionDecodingError(response, str(error)) from error
        return DecisionResponse(output=output, raw=response)
