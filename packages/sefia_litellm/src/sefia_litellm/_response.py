from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Coroutine, cast

from sefia.llm import LLMResponse, ToolCall
from sefia.llm.streaming import OutputStreamCallback

from ._schema import DecisionEnvelopeFormat
from ._output_stream import OutputEventStreamer

if TYPE_CHECKING:
    from litellm import Choices, ModelResponse, Usage

logger = logging.getLogger(__name__)


def handle_response(
    response: ModelResponse,
    *,
    requested_model: str,
    output: DecisionEnvelopeFormat | None,
) -> LLMResponse:
    if not response.choices:
        raise RuntimeError(
            f"LLM returned empty choices (model={requested_model}). "
            "This may indicate a content filter, provider error, or a LiteLLM bug."
        )

    choice: Choices = response.choices[0]
    message = choice.message
    tool_calls = [_function_tool_call(call) for call in (message.tool_calls or [])]
    usage = cast("Usage | None", cast(dict[str, Any], response).get("usage"))
    result = LLMResponse(
        model=response.model,
        content=message.content,
        reasoning_content=getattr(message, "reasoning_content", None),
        tool_calls=tool_calls,
        usage=usage.model_dump() if usage else None,
        stop_reason=choice.finish_reason,
        cost=_calculate_cost(response),
    )
    _decode_output(result, output)
    return result


def _function_tool_call(call: Any) -> ToolCall:
    function = getattr(call, "function", None)
    if function is None:
        raise RuntimeError("LLM returned an unsupported custom tool call")
    return ToolCall(id=call.id, function=function.model_dump())


async def handle_stream(
    stream: AsyncIterator[Any],
    *,
    content_callback: Callable[[str], Coroutine[None, None, None]] | None,
    output_callback: OutputStreamCallback | None,
    reasoning_callback: Callable[[str], Coroutine[None, None, None]] | None,
    messages: list[dict[str, Any]],
    output: DecisionEnvelopeFormat | None,
    requested_model: str,
) -> LLMResponse:
    import litellm
    from litellm import ModelResponse

    chunks: list[Any] = []
    reasoning_parts: list[str] = []
    event_streamer = (
        OutputEventStreamer(output, output_callback)
        if output is not None and output_callback is not None
        else None
    )
    async for chunk in stream:
        chunks.append(chunk)
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_parts.append(reasoning)
            if reasoning_callback is not None:
                await reasoning_callback(reasoning)
        content = getattr(delta, "content", None)
        if content:
            if content_callback is not None:
                await content_callback(content)
            if event_streamer is not None:
                await event_streamer.feed(content)

    build_stream_response = cast(
        Callable[..., ModelResponse | None],
        getattr(cast(object, litellm), "stream_chunk_builder"),
    )
    response = build_stream_response(chunks=chunks, messages=messages)
    if not isinstance(response, ModelResponse):
        raise RuntimeError("Invalid model response")

    result = handle_response(
        response,
        requested_model=requested_model,
        output=output,
    )
    if reasoning_parts and result.reasoning_content is None:
        result.reasoning_content = "".join(reasoning_parts)
    return result


def _decode_output(
    response: LLMResponse, output: DecisionEnvelopeFormat | None
) -> None:
    if output is None or response.content is None:
        return
    raw = response.content.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]).strip()
    try:
        response.structured_output = output.decode_json(raw)
    except ValueError:
        return


def _calculate_cost(response: ModelResponse) -> float | None:
    from litellm import cost_per_token

    usage = cast("Usage | None", cast(dict[str, Any], response).get("usage"))
    model = response.model
    if not (usage and model):
        return None
    try:
        prompt_cost, completion_cost = cost_per_token(
            model=model,
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
        )
        return prompt_cost + completion_cost
    except Exception:
        logger.warning("Failed to calculate cost for model %s", model, exc_info=True)
        return None


__all__ = ["handle_response", "handle_stream"]
