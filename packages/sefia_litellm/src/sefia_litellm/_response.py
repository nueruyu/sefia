from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Coroutine, cast

from sefia.llm import LLMOutput, LLMResponse, LLMResponseDecodingError, ToolCall
from sefia.llm.streaming import (
    JsonOutputStreamDecoder,
    OutputStreamCallback,
    OutputStreamEvent,
    Scalar,
    StringDelta,
    StringEnd,
)

from ._schema import StructuredDecisionFormat, StructuredValueFormat

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
    tool_argument_formats: dict[str, StructuredValueFormat] | None = None,
) -> LLMResponse:
    if not response.choices:
        raise RuntimeError(
            f"LLM returned empty choices (model={requested_model}). "
            "This may indicate a content filter, provider error, or a LiteLLM bug."
        )

    choice: Choices = response.choices[0]
    message = choice.message
    usage = cast("Usage | None", cast(dict[str, Any], response).get("usage"))
    result = LLMResponse(
        model=response.model,
        content=message.content,
        reasoning_content=getattr(message, "reasoning_content", None),
        usage=usage.model_dump() if usage else None,
        stop_reason=choice.finish_reason,
        cost=_calculate_cost(response),
    )
    try:
        argument_formats = tool_argument_formats or {}
        result.tool_calls = [
            _function_tool_call(call, argument_formats)
            for call in (message.tool_calls or [])
        ]
    except (RuntimeError, ValueError) as error:
        raise LLMResponseDecodingError(result, str(error)) from error
    _decode_output(result, output)
    return result


def _function_tool_call(
    call: ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall,
    tool_argument_formats: dict[str, StructuredValueFormat],
) -> ToolCall:
    if getattr(call, "type", None) == "custom":
        raise RuntimeError("LLM returned an unsupported custom tool call")
    function_call = cast("ChatCompletionMessageToolCall", call)
    name = function_call.function.name
    if not isinstance(name, str) or not name:
        raise ValueError("Native tool call has no function name.")
    arguments_json = cast(object, function_call.function.arguments)
    if not isinstance(arguments_json, str):
        raise ValueError(f"Native tool call {name!r} has no JSON arguments.")
    arguments = LLMOutput.parse_json(arguments_json)
    value_format = tool_argument_formats.get(name)
    if value_format is not None:
        arguments = value_format.decode(arguments)
    return ToolCall(id=function_call.id, name=name, arguments=arguments)


async def handle_stream(
    stream: AsyncIterator[Any],
    *,
    content_callback: Callable[[str], Coroutine[None, None, None]] | None,
    output_callback: OutputStreamCallback | None,
    reasoning_callback: Callable[[str], Coroutine[None, None, None]] | None,
    messages: list[dict[str, Any]],
    output: StructuredDecisionFormat | None,
    tool_argument_formats: dict[str, StructuredValueFormat] | None = None,
    requested_model: str,
) -> LLMResponse:
    import litellm
    from litellm import ModelResponse

    chunks: list[Any] = []
    events = _StreamEventDispatcher(
        content_callback=content_callback,
        output_callback=output_callback,
        reasoning_callback=reasoning_callback,
        structured=output is not None,
        tool_argument_formats=tool_argument_formats or {},
    )
    async for chunk in stream:
        chunks.append(chunk)
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        await events.feed(getattr(choices[0], "delta", None))

    await events.finish()

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
        tool_argument_formats=tool_argument_formats,
    )
    if events.reasoning_text and result.reasoning_content is None:
        result.reasoning_content = events.reasoning_text
    return result


class _StreamEventDispatcher:
    def __init__(
        self,
        *,
        content_callback: Callable[[str], Coroutine[None, None, None]] | None,
        output_callback: OutputStreamCallback | None,
        reasoning_callback: Callable[[str], Coroutine[None, None, None]] | None,
        structured: bool,
        tool_argument_formats: dict[str, StructuredValueFormat],
    ) -> None:
        self._content_callback = content_callback
        self._output_callback = output_callback
        self._reasoning_callback = reasoning_callback
        self._reasoning_parts: list[str] = []
        self._structured_decoder = (
            JsonOutputStreamDecoder()
            if structured and output_callback is not None
            else None
        )
        self._native_decoder = (
            _NativeToolStreamDecoder(tool_argument_formats)
            if tool_argument_formats and output_callback is not None
            else None
        )

    @property
    def reasoning_text(self) -> str:
        return "".join(self._reasoning_parts)

    async def feed(self, delta: object) -> None:
        reasoning = getattr(delta, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning:
            self._reasoning_parts.append(reasoning)
            if self._reasoning_callback is not None:
                await self._reasoning_callback(reasoning)

        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            if self._content_callback is not None:
                await self._content_callback(content)
            if self._structured_decoder is not None:
                await self._emit(self._structured_decoder.feed(content))

        if self._native_decoder is not None:
            fragments = _extract_native_tool_call_fragments(
                getattr(delta, "tool_calls", None)
            )
            await self._emit(self._native_decoder.feed(fragments))

    async def finish(self) -> None:
        if self._native_decoder is not None:
            await self._emit(self._native_decoder.finish())

    async def _emit(self, events: list[OutputStreamEvent]) -> None:
        assert self._output_callback is not None
        for event in events:
            await self._output_callback(event)


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


class _NativeToolStreamDecoder:
    def __init__(
        self,
        tool_argument_formats: dict[str, StructuredValueFormat],
    ) -> None:
        self._tool_argument_formats = tool_argument_formats
        self._names: dict[int, str] = {}
        self._announced: set[int] = set()
        self._arguments: dict[int, str] = {}
        self._consumed: dict[int, int] = {}
        self._decoders: dict[int, JsonOutputStreamDecoder] = {}

    def feed(
        self,
        fragments: list[_NativeToolCallFragment],
    ) -> list[OutputStreamEvent]:
        events: list[OutputStreamEvent] = []
        for fragment in fragments:
            if fragment.name is not None:
                self._names[fragment.index] = fragment.name
                if fragment.index not in self._announced:
                    self._announced.add(fragment.index)
                    events.append(
                        StringEnd(
                            ("tool_calls", fragment.index, "name"),
                            fragment.name,
                        )
                    )
            if fragment.arguments_json:
                self._arguments[fragment.index] = (
                    self._arguments.get(fragment.index, "") + fragment.arguments_json
                )
            events.extend(self._decode_available(fragment.index))
        return events

    def finish(self) -> list[OutputStreamEvent]:
        events: list[OutputStreamEvent] = []
        for index, arguments in self._arguments.items():
            name = self._names.get(index)
            value_format = (
                self._tool_argument_formats.get(name) if name is not None else None
            )
            if value_format is None or not value_format.translates_values:
                events.extend(self._decode_available(index))
                continue
            try:
                output = value_format.decode(LLMOutput.parse_json(arguments))
            except ValueError:
                continue
            events.extend(
                _logical_output_events(
                    output,
                    ("tool_calls", index, "arguments"),
                )
            )
        return events

    def _decode_available(self, index: int) -> list[OutputStreamEvent]:
        name = self._names.get(index)
        if name is None:
            return []
        value_format = self._tool_argument_formats.get(name)
        if value_format is not None and value_format.translates_values:
            return []

        arguments = self._arguments.get(index, "")
        consumed = self._consumed.get(index, 0)
        token = arguments[consumed:]
        if not token:
            return []
        self._consumed[index] = len(arguments)
        decoder = self._decoders.setdefault(index, JsonOutputStreamDecoder())
        return [_tool_argument_event(index, event) for event in decoder.feed(token)]


def _logical_output_events(
    output: LLMOutput,
    path: tuple[str | int, ...],
) -> list[OutputStreamEvent]:
    value = output.data
    if isinstance(value, str):
        return [StringEnd(path, value)]
    if value is None or isinstance(value, int | float | bool):
        return [Scalar(path, value)]
    if isinstance(value, list):
        return [
            event
            for index, item in enumerate(value)
            for event in _logical_output_events(
                LLMOutput.from_data(item),
                (*path, index),
            )
        ]
    return [
        event
        for name, item in value.items()
        for event in _logical_output_events(
            LLMOutput.from_data(item),
            (*path, name if isinstance(name, str | int) else str(name)),
        )
    ]


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
