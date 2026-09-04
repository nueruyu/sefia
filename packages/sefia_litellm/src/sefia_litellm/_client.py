from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable
from typing import Any, Coroutine, cast

from typing_extensions import final, override

from sefia.exceptions import InferenceError
from sefia.llm import LLMClient, LLMResponse, Message
from sefia.llm.step_decision import DecisionSpec, StepTool
from sefia.llm.streaming import OutputStreamCallback

from ._request import build_completion_request
from ._response import handle_response, handle_stream
from .exceptions import (
    InferenceConnectionError,
    InferenceRateLimitError,
    InferenceTemporarilyUnavailableError,
    InferenceTimeoutError,
)

# Prevent LiteLLM from fetching its cost map during lazy import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_LOG_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_SILENCE_LEVEL = logging.CRITICAL + 1  # Disable every LiteLLM log record.


def _env_suppress_logs_default() -> bool:
    raw = os.environ.get("SEFIA_LITELLM_SUPPRESS_LOGS", "").strip().lower()
    return raw not in _LOG_FALSE_VALUES


def _apply_litellm_log_level(suppress: bool) -> None:
    logging.getLogger("LiteLLM").setLevel(
        _SILENCE_LEVEL if suppress else logging.NOTSET
    )


def _configure_litellm_logging(suppress: bool) -> None:
    import litellm

    _apply_litellm_log_level(suppress)
    litellm.suppress_debug_info = suppress


_apply_litellm_log_level(_env_suppress_logs_default())


def _to_inference_error(error: Exception) -> InferenceError | None:
    from litellm.exceptions import (
        APIConnectionError,
        InternalServerError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
    )

    # Timeout subclasses APIConnectionError, so ordering is significant.
    error_mapping: tuple[tuple[type[Exception], type[InferenceError]], ...] = (
        (Timeout, InferenceTimeoutError),
        (APIConnectionError, InferenceConnectionError),
        (RateLimitError, InferenceRateLimitError),
        (InternalServerError, InferenceTemporarilyUnavailableError),
        (ServiceUnavailableError, InferenceTemporarilyUnavailableError),
    )
    for provider_error, inference_error in error_mapping:
        if isinstance(error, provider_error):
            return inference_error(str(error))
    return None


@final
class LiteLLMClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        suppress_logs: bool | None = None,
        **kwargs: Any,
    ):
        if "native_structured_output" in kwargs:
            raise TypeError(
                "native_structured_output is no longer supported; select "
                "PromptedDecisionTransport for models without structured output."
            )
        self.model = model
        self._kwargs = kwargs
        self._suppress_logs = (
            _env_suppress_logs_default() if suppress_logs is None else suppress_logs
        )

    @override
    async def complete(
        self,
        messages: list[Message],
        tools: list[StepTool] | None = None,
        decision_model: DecisionSpec | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
        output_callback: OutputStreamCallback | None = None,
        reasoning_callback: (
            Callable[[str], Coroutine[None, None, None]] | None
        ) = None,
    ) -> LLMResponse:
        import litellm
        from litellm import ModelResponse

        _configure_litellm_logging(self._suppress_logs)
        request = build_completion_request(
            messages=messages,
            tools=tools,
            decision_model=decision_model,
            client_kwargs=self._kwargs,
            stream=any(
                callback is not None
                for callback in (
                    stream_callback,
                    output_callback,
                    reasoning_callback,
                )
            ),
        )

        try:
            complete = cast(
                Callable[..., Coroutine[Any, Any, Any]],
                getattr(cast(object, litellm), "acompletion"),
            )
            response = await complete(
                model=self.model,
                messages=request.messages,
                **request.kwargs,
            )
            if hasattr(response, "__aiter__") and not isinstance(
                response, ModelResponse
            ):
                return await handle_stream(
                    cast(AsyncIterator[Any], response),
                    content_callback=stream_callback,
                    output_callback=output_callback,
                    reasoning_callback=reasoning_callback,
                    messages=request.messages,
                    output=request.decision_format,
                    tool_argument_formats=request.tool_argument_formats,
                    requested_model=self.model,
                )
        except Exception as error:
            inference_error = _to_inference_error(error)
            if inference_error is not None:
                raise inference_error from error
            raise

        if not isinstance(response, ModelResponse):
            raise RuntimeError("Invalid model response")
        return handle_response(
            response,
            requested_model=self.model,
            output=request.decision_format,
            tool_argument_formats=request.tool_argument_formats,
        )


__all__ = ["LiteLLMClient"]
