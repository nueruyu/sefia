from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Coroutine, cast

from sefia.llm import LLMResponse, ToolCall
from sefia.llm.streaming import (
    JsonOutputStreamDecoder,
    OutputStreamCallback,
    OutputStreamEvent,
    Scalar,
    StringDelta,
    StringEnd,
)

from ._schema import StructuredDecisionFormat

if TYPE_CHECKING:
    from litellm import Choices, ModelResponse, Usage
    from litellm.types.utils import (  # pyright: ignore[reportMissingTypeStubs]
        ChatCompletionMessageCustomToolCall,
        ChatCompletionMessageToolCall,
    )

logger = logging.getLogger(__name__)


def handle_response(
    response: ModelResponse,
    *,
    requested_model: str,
    output: StructuredDecisionFormat | None,
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


def _function_tool_call(
    call: ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall,
) -> ToolCall:
    if getattr(call, "type", None) == "custom":
        raise RuntimeError("LLM returned an unsupported custom tool call")
    function_call = cast("ChatCompletionMessageToolCall", call)
    return ToolCall(
        id=function_call.id,
        function=function_call.function.model_dump(),
    )


async def handle_stream(
    stream: AsyncIterator[Any],
    *,
    content_callback: Callable[[str], Coroutine[None, None, None]] | None,
    output_callback: OutputStreamCallback | None,
    reasoning_callback: Callable[[str], Coroutine[None, None, None]] | None,
    messages: list[dict[str, Any]],
    output: StructuredDecisionFormat | None,
    requested_model: str,
) -> LLMResponse:
    import litellm
    from litellm import ModelResponse

    chunks: list[Any] = []
    reasoning_parts: list[str] = []
    native_argument_decoders: dict[int, JsonOutputStreamDecoder] = {}
    identified_native_tools: set[int] = set()
    event_decoder = (
        JsonOutputStreamDecoder()
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
            if event_decoder is not None:
                assert output_callback is not None
                for event in event_decoder.feed(content):
                    await output_callback(event)
        if output is None and output_callback is not None:
            for fragment in _extract_native_tool_call_fragments(
                getattr(delta, "tool_calls", None)
            ):
                if (
                    fragment.name is not None
                    and fragment.index not in identified_native_tools
                ):
                    identified_native_tools.add(fragment.index)
                    await output_callback(
                        StringEnd(("tool_calls", fragment.index, "name"), fragment.name)
                    )
                if fragment.arguments_json:
                    decoder = native_argument_decoders.get(fragment.index)
                    if decoder is None:
                        decoder = JsonOutputStreamDecoder()
                        native_argument_decoders[fragment.index] = decoder
                    for event in decoder.feed(fragment.arguments_json):
                        await output_callback(
                            _tool_argument_event(fragment.index, event)
                        )

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


@dataclass(frozen=True)
class _NativeToolCallFragment:
    index: int
    name: str | None
    arguments_json: str | None


def _extract_native_tool_call_fragments(
    calls: object,
) -> list[_NativeToolCallFragment]:
    if not isinstance(calls, list):
        return []

    fragments: list[_NativeToolCallFragment] = []
    for call in cast(list[object], calls):
        index = getattr(call, "index", None)
        function = getattr(call, "function", None)
        if not isinstance(index, int) or function is None:
            continue
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        fragments.append(
            _NativeToolCallFragment(
                index=index,
                name=name if isinstance(name, str) and name else None,
                arguments_json=(
                    arguments if isinstance(arguments, str) and arguments else None
                ),
            )
        )
    return fragments


def _tool_argument_event(index: int, event: OutputStreamEvent) -> OutputStreamEvent:
    path = ("tool_calls", index, "arguments", *event.path)
    if isinstance(event, StringDelta):
        return StringDelta(path, event.text)
    if isinstance(event, StringEnd):
        return StringEnd(path, event.value)
    assert isinstance(event, Scalar)
    return Scalar(path, event.value)


def _decode_output(
    response: LLMResponse, output: StructuredDecisionFormat | None
) -> None:
    if output is None or response.content is None:
        return
    try:
        response.structured_output = output.decode_json(response.content)
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
