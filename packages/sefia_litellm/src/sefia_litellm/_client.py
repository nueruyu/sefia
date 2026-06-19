from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Callable, Coroutine, cast

from sefia.exceptions import (
    ConnectionException,
    InferenceException,
    RateLimitException,
    TemporarilyUnavailableException,
    TimeoutException,
)
from sefia.llm import LLMClient, LLMResponse, Message, ToolCall

if TYPE_CHECKING:
    from litellm import Choices, ModelResponse, Usage

logger = logging.getLogger(__name__)

# Set before litellm is imported (it is imported lazily, well after this module
# loads). Forces litellm to use its bundled model cost map instead of fetching it
# from the network at import time, which speeds up the import and keeps it working
# offline. A user who has already set this explicitly is respected.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# LiteLLM is imported lazily (inside the methods that need it) because importing
# it eagerly is slow and would penalize anyone who merely imports
# ``sefia_litellm`` without making a request. After the first call the module is
# cached in ``sys.modules``, so subsequent local imports are effectively free.

_LOG_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

# A level above CRITICAL drops every record, fully silencing the logger. LiteLLM
# surfaces real failures as exceptions (see ``_to_inference_exception``), so its
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
    """Sets the ``"LiteLLM"`` logger level.

    Does not import litellm, so it can run before litellm is imported — that way
    even litellm's import-time warnings (e.g. optional-dependency preload
    warnings) are suppressed. ``suppress`` fully silences the logger; otherwise
    the level is reset to ``NOTSET`` (inherit from the root logger).
    """
    logging.getLogger("LiteLLM").setLevel(
        _SILENCE_LEVEL if suppress else logging.NOTSET
    )


def _configure_litellm_logging(suppress: bool) -> None:
    """Applies the full logging configuration once LiteLLM is imported.

    Called from ``complete()``. Sets the logger level and toggles
    ``suppress_debug_info`` (which controls the "Provider List: ..." banner and
    the debug info printed alongside exceptions). Both are process-global, so with
    multiple clients the last ``complete()`` call wins.
    """
    import litellm

    _apply_litellm_log_level(suppress)
    litellm.suppress_debug_info = suppress


# Silence LiteLLM as early as possible — before it is imported (lazily, on the
# first request) — so its import-time warnings are suppressed too when the
# resolved default is "suppress". An explicit per-client ``suppress_logs`` still
# takes effect later via ``_configure_litellm_logging``.
_apply_litellm_log_level(_env_suppress_logs_default())


def _to_inference_exception(error: Exception) -> InferenceException | None:
    """Maps a LiteLLM exception to a sefia InferenceException, if recognized."""
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

    exception_mapping: tuple[tuple[type[Exception], type[InferenceException]], ...] = (
        (Timeout, TimeoutException),
        (APIConnectionError, ConnectionException),
        (RateLimitError, RateLimitException),
        (InternalServerError, TemporarilyUnavailableException),
        (ServiceUnavailableError, TemporarilyUnavailableException),
    )

    for provider_exc, inference_exc in exception_mapping:
        if isinstance(error, provider_exc):
            return inference_exc(str(error))
    return None


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

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        output_schema: dict | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
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
        from litellm import ModelResponse, stream_chunk_builder

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
            tool_calls=tool_calls,
            usage=usage.model_dump() if usage else None,
            stop_reason=choice.finish_reason,
            cost=cost,
        )
