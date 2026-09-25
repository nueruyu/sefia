from typing_extensions import final, override

from .._client import LLMClient
from .._json import JsonSnapshot
from .._prompt_renderer import PromptRenderer
from .._prompted_response import PromptedJsonStreamExtractor, extract_prompted_json
from ..exceptions import DecisionDecodingError
from ..streaming import JsonOutputStreamDecoder
from ._base import (
    DecisionObserver,
    DecisionRequest,
    DecisionTransport,
    DecodedDecision,
)
from ._decision_instructions import json_response_instructions
from ._messages import build_text_decision_messages


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
    ) -> DecodedDecision:
        messages = build_text_decision_messages(
            request=request,
            renderer=prompt_renderer,
            response_instructions=json_response_instructions(request.decision_spec),
        )
        stream_decoder = JsonOutputStreamDecoder() if stream else None
        extractor = PromptedJsonStreamExtractor() if stream else None

        async def on_text(text: str) -> None:
            assert stream_decoder is not None and extractor is not None
            await observer.response_text(text)
            json_text = extractor.feed(text)
            if json_text:
                for event in stream_decoder.feed(json_text):
                    await observer.output(event)

        await observer.before_request(tuple(messages))
        completion = await client.complete(
            messages=messages,
            tools=None,
            decision_spec=None,
            stream_callback=on_text if stream else None,
            output_callback=None,
            reasoning_callback=observer.reasoning_text if stream else None,
        )
        if completion.content is None:
            raise DecisionDecodingError(
                completion, "LLM did not provide response content."
            )
        try:
            data = JsonSnapshot.parse_json(extract_prompted_json(completion.content))
        except ValueError as error:
            raise DecisionDecodingError(completion, str(error)) from error
        return DecodedDecision(decision_data=data, completion=completion)
