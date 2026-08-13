from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Callable, Coroutine, cast

from typing_extensions import final, override

from sefia.exceptions import InferenceError
from sefia.llm import LLMClient, LLMResponse, Message, ToolCall

from .exceptions import (
    InferenceConnectionError,
    InferenceRateLimitError,
    InferenceTemporarilyUnavailableError,
    InferenceTimeoutError,
)

if TYPE_CHECKING:
    from litellm import Choices, ModelResponse, Usage

logger = logging.getLogger(__name__)

# Set before litellm is imported (it is imported lazily, well after this module
# loads). Forces litellm to use its bundled model cost map instead of fetching it
# from the network at import time, which speeds up the import and keeps it working
# offline. A user who has already set this explicitly is respected.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_LOG_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

# A level above CRITICAL drops every record, fully silencing the logger. LiteLLM
# surfaces real failures as exceptions (see ``_to_inference_error``), so its
# log output is noise that can be turned off entirely without hiding errors.
_SILENCE_LEVEL = logging.CRITICAL + 1


def _env_suppress_logs_default() -> bool:
    """Resolves the default for ``suppress_logs`` from the environment.

    Reads ``SEFIA_LITELLM_SUPPRESS_LOGS``; when unset or empty, logs are
    suppressed by default. Any of ``0/false/no/off`` (case-insensitive) disables
    suppression.
    """
    raw = os.environ.get("SEFIA_LITELLM_SUPPRESS_LOGS", "").strip().lower()
    return raw not in _LOG_FALSE_VALUES


def _apply_litellm_log_level(suppress: bool) -> None:
    """Set the LiteLLM logger without importing LiteLLM."""
    logging.getLogger("LiteLLM").setLevel(
        _SILENCE_LEVEL if suppress else logging.NOTSET
    )


def _configure_litellm_logging(suppress: bool) -> None:
    """Apply process-global LiteLLM logging; the last client call wins."""
    import litellm

    _apply_litellm_log_level(suppress)
    litellm.suppress_debug_info = suppress


# Silence LiteLLM as early as possible — before it is imported (lazily, on the
# first request) — so its import-time warnings are suppressed too when the
# resolved default is "suppress". An explicit per-client ``suppress_logs`` still
# takes effect later via ``_configure_litellm_logging``.
_apply_litellm_log_level(_env_suppress_logs_default())


def _to_inference_error(error: Exception) -> InferenceError | None:
    """Maps a LiteLLM exception to a sefia InferenceError, if recognized."""
    # Imported here rather than at module load to keep LiteLLM out of the import
    # path; this is only reached after a request has already imported it. Order
    # matters: Timeout subclasses APIConnectionError, so it must be checked
    # first. Errors not listed here (AuthenticationError, BadRequestError,
    # ContextWindowExceededError, ContentPolicyViolationError, ...) are
    # deterministic and propagate unchanged as genuine failures.
    from litellm.exceptions import (
        APIConnectionError,
        InternalServerError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
    )

    error_mapping: tuple[tuple[type[Exception], type[InferenceError]], ...] = (
        (Timeout, InferenceTimeoutError),
        (APIConnectionError, InferenceConnectionError),
        (RateLimitError, InferenceRateLimitError),
        (InternalServerError, InferenceTemporarilyUnavailableError),
        (ServiceUnavailableError, InferenceTemporarilyUnavailableError),
    )

    for provider_exc, inference_error in error_mapping:
        if isinstance(error, provider_exc):
            return inference_error(str(error))
    return None


@final
class LiteLLMClient(LLMClient):
    """
    An LLMClient implementation that uses LiteLLM to interact with
    various LLM providers.
    """

    def __init__(self, model: str, *, suppress_logs: bool | None = None, **kwargs: Any):
        self.model = model
        self._kwargs = kwargs
        # ``None`` defers to SEFIA_LITELLM_SUPPRESS_LOGS (default: suppress);
        # an explicit bool overrides the environment.
        self._suppress_logs = (
            _env_suppress_logs_default() if suppress_logs is None else suppress_logs
        )

    @override
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        output_schema: dict | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
        reasoning_callback: (
            Callable[[str], Coroutine[None, None, None]] | None
        ) = None,
    ) -> LLMResponse:
        """Sends a completion request using LiteLLM."""
        from litellm import ModelResponse, acompletion

        _configure_litellm_logging(self._suppress_logs)

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

        if stream_callback or reasoning_callback:
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
                return await self._handle_stream(
                    stream, stream_callback, reasoning_callback, raw_messages
                )
        except Exception as e:
            inference_error = _to_inference_error(e)
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
        reasoning_callback: Callable[[str], Coroutine[None, None, None]] | None,
        raw_messages: list[dict[str, Any]],
    ) -> LLMResponse:
        """Processes a streaming response."""
        from litellm import ModelResponse, stream_chunk_builder

        chunks = []
        # ``stream_chunk_builder`` does not reliably reassemble reasoning content,
        # so we accumulate reasoning deltas ourselves and attach them to the final
        # response below.
        reasoning_parts: list[str] = []
        async for chunk in stream:
            chunks.append(chunk)
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
                if reasoning_callback:
                    await reasoning_callback(reasoning)
            content = getattr(delta, "content", None)
            if content and callback:
                await callback(content)

        response = stream_chunk_builder(chunks=chunks, messages=raw_messages)
        if not isinstance(response, ModelResponse):
            raise RuntimeError("Invalid model response")

        result = self._handle_response(response)
        if reasoning_parts and result.reasoning_content is None:
            result.reasoning_content = "".join(reasoning_parts)
        return result

    def _calculate_cost(self, response: ModelResponse) -> float | None:
        """Calculates the cost of a response, if usage data is available."""
        from litellm import cost_per_token

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
            reasoning_content=getattr(message, "reasoning_content", None),
            tool_calls=tool_calls,
            usage=usage.model_dump() if usage else None,
            stop_reason=choice.finish_reason,
            cost=cost,
        )
