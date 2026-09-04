from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Coroutine, cast

from sefia.llm import LLMOutput, LLMResponse
from sefia.llm.exceptions import LLMResponseDecodingError
from sefia.llm.streaming import (
    JsonOutputStreamDecoder,
    OutputStreamCallback,
    OutputStreamEvent,
    Scalar,
    StringDelta,
    StringEnd,
)

from ._response import handle_response
from ._schema import StructuredDecisionFormat, StructuredValueFormat


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
        raise LLMResponseDecodingError(
            LLMResponse(
                content=events.content_text or None,
                reasoning_content=events.reasoning_text or None,
            ),
            "LiteLLM could not reconstruct a model response from the stream.",
        )

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
        self._content_parts: list[str] = []
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

    @property
    def content_text(self) -> str:
        return "".join(self._content_parts)

    async def feed(self, delta: object) -> None:
        reasoning = getattr(delta, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning:
            self._reasoning_parts.append(reasoning)
            if self._reasoning_callback is not None:
                await self._reasoning_callback(reasoning)

        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            self._content_parts.append(content)
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


__all__ = ["handle_stream"]
