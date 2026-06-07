import logging
from collections.abc import AsyncIterator
from typing import Any, Callable, Coroutine, cast

from litellm import (
    Choices,
    ModelResponse,
    Usage,
    acompletion,
    cost_per_token,
    stream_chunk_builder,
)
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from sefia.exceptions import (
    ConnectionException,
    InferenceException,
    RateLimitException,
    TemporarilyUnavailableException,
    TimeoutException,
)
from sefia.llm.client import LLMClient
from sefia.llm.messages import LLMResponse, Message, ToolCall

logger = logging.getLogger(__name__)

# Translates provider-specific exceptions into sefia's abstract inference
# exceptions, so callers never have to know about LiteLLM's types. Order
# matters: Timeout subclasses APIConnectionError, so it must be checked first.
# Errors not listed here (AuthenticationError, BadRequestError,
# ContextWindowExceededError, ContentPolicyViolationError, ...) are deterministic
# and propagate unchanged as genuine failures.
_EXCEPTION_MAPPING: tuple[tuple[type[Exception], type[InferenceException]], ...] = (
    (Timeout, TimeoutException),
    (APIConnectionError, ConnectionException),
    (RateLimitError, RateLimitException),
    (InternalServerError, TemporarilyUnavailableException),
    (ServiceUnavailableError, TemporarilyUnavailableException),
)


def _to_inference_exception(error: Exception) -> InferenceException | None:
    """Maps a LiteLLM exception to a sefia InferenceException, if recognized."""
    for provider_exc, inference_exc in _EXCEPTION_MAPPING:
        if isinstance(error, provider_exc):
            return inference_exc(str(error))
    return None


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
        raw_messages = [msg.to_dict(exclude_none=True) for msg in messages]

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

        try:
            response = await acompletion(
                model=self.model,
                messages=raw_messages,
                **kwargs,
            )

            if hasattr(response, "__aiter__") and not isinstance(
                response, ModelResponse
            ):
                stream = cast(AsyncIterator[Any], response)
                return await self._handle_stream(stream, stream_callback, raw_messages)
        except Exception as e:
            inference_error = _to_inference_exception(e)
            if inference_error is not None:
                raise inference_error from e
            raise

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

    def _calculate_cost(self, response: ModelResponse) -> float | None:
        """Calculates the cost of a response, if usage data is available."""
        usage: Usage | None = response.get("usage")
        model = response.model

        if not (usage and model):
            return None
        try:
            prompt_tokens = usage.prompt_tokens or 0
            completion_tokens = usage.completion_tokens or 0
            prompt_cost, completion_cost = cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return prompt_cost + completion_cost
        except Exception:
            logger.warning(
                "Failed to calculate cost for model %s",
                model,
                exc_info=True,
            )
            return None

    def _handle_response(self, response: ModelResponse) -> LLMResponse:
        """Processes a non-streaming or completed stream response."""
        if not response.choices:
            raise RuntimeError(
                f"LLM returned empty choices (model={self.model}). "
                "This may indicate a content filter, provider error, or a LiteLLM bug."
            )

        choice: Choices = response.choices[0]
        message = choice.message

        tool_calls = [
            ToolCall(
                id=tc.id,
                function=tc.function.model_dump(),
            )
            for tc in (message.tool_calls or [])
        ]

        usage: Usage | None = response.get("usage")
        cost = self._calculate_cost(response)

        return LLMResponse(
            model=response.model,
            content=message.content,
            tool_calls=tool_calls,
            usage=usage.model_dump() if usage else None,
            stop_reason=choice.finish_reason,
            cost=cost,
        )
