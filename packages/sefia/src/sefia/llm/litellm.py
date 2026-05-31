from collections.abc import AsyncIterator
from typing import Any, Callable, Coroutine, cast

from litellm import ModelResponse, Usage, acompletion, stream_chunk_builder

from sefia.llm.client import LLMClient

from .messages import LLMResponse, Message, ToolCall


class LiteLLMClient(LLMClient):
    """
    An LLMClient implementation that uses LiteLLM to interact with
    various LLM providers.
    """

    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self._kwargs = kwargs

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        output_schema: dict | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
    ) -> LLMResponse:
        """Sends a completion request using LiteLLM."""
        raw_messages = [msg.model_dump(exclude_none=True) for msg in messages]

        kwargs = self._kwargs.copy()
        if tools:
            kwargs["tools"] = tools
        if output_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": output_schema,
                    "strict": True,
                },
            }

        if stream_callback:
            kwargs["stream"] = True

        response = await acompletion(
            model=self.model,
            messages=raw_messages,
            **kwargs,
        )

        if hasattr(response, "__aiter__") and not isinstance(response, ModelResponse):
            stream = cast(AsyncIterator[Any], response)
            return await self._handle_stream(stream, stream_callback, raw_messages)

        if not isinstance(response, ModelResponse):
            raise RuntimeError("Invalid model response")
        return self._handle_response(response)

    async def _handle_stream(
        self,
        stream: AsyncIterator[Any],
        callback: Callable[[str], Coroutine[None, None, None]] | None,
        raw_messages: list[dict[str, Any]],
    ) -> LLMResponse:
        """Processes a streaming response."""
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
            choices = getattr(chunk, "choices", None)
            if callback and choices:
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None)
                if content:
                    await callback(content)

        response = stream_chunk_builder(chunks=chunks, messages=raw_messages)
        if not isinstance(response, ModelResponse):
            raise RuntimeError("Invalid model response")

        return self._handle_response(response)

    def _handle_response(self, response: ModelResponse) -> LLMResponse:
        """Processes a non-streaming or completed stream response."""
        if not response.choices:
            raise RuntimeError(
                f"LLM returned empty choices (model={self.model}). "
                "This may indicate a content filter, provider error, or a LiteLLM bug."
            )

        choice = response.choices[0]
        message = choice.message

        tool_calls = [
            ToolCall(
                id=tc.id,
                function=tc.function.model_dump(),
            )
            for tc in (message.tool_calls or [])
        ]

        usage: Usage | None = response.get("usage")

        return LLMResponse(
            model=response.model,
            content=message.content,
            tool_calls=tool_calls,
            usage=usage.model_dump() if usage else None,
            stop_reason=choice.finish_reason,
        )
