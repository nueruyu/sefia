from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Coroutine, Protocol, cast

from typing_extensions import final

from sefia.llm import LLMCompletion
from sefia.llm.exceptions import LLMCompletionDecodingError
from sefia.llm.streaming import (
    JsonOutputStreamDecoder,
    OutputStreamCallback,
    OutputStreamEvent,
)

from ._native_tool_stream import NativeToolCallDelta, NativeToolCallStreamDecoder
from ._response import decode_completion
from ._schema import StructuredDecisionFormat
from ._schema._data_format import StructuredDataFormat


class _CompletionDelta(Protocol):
    content: str | None
    reasoning_content: str | None
    tool_calls: list[NativeToolCallDelta] | None


async def consume_completion_stream(
    stream: AsyncIterator[Any],
    *,
    content_callback: Callable[[str], Coroutine[None, None, None]] | None,
    output_callback: OutputStreamCallback | None,
    reasoning_callback: Callable[[str], Coroutine[None, None, None]] | None,
    messages: list[dict[str, Any]],
    decision_format: StructuredDecisionFormat | None,
    tool_data_formats: dict[str, StructuredDataFormat] | None = None,
    requested_model: str,
) -> LLMCompletion:
    import litellm
    from litellm import ModelResponse

    chunks: list[Any] = []
    state = _CompletionStreamState(
        content_callback=content_callback,
        output_callback=output_callback,
        reasoning_callback=reasoning_callback,
        decision_format=decision_format,
        tool_data_formats=tool_data_formats or {},
    )
    async for chunk in stream:
        chunks.append(chunk)
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        await state.feed(
            cast(_CompletionDelta | None, getattr(choices[0], "delta", None))
        )

    await state.finish()

    build_stream_response = cast(
        Callable[..., ModelResponse | None],
        getattr(cast(object, litellm), "stream_chunk_builder"),
    )
    response = build_stream_response(chunks=chunks, messages=messages)
    if not isinstance(response, ModelResponse):
        raise LLMCompletionDecodingError(
            LLMCompletion(
                content=state.content_text or None,
                reasoning_content=state.reasoning_text or None,
            ),
            "LiteLLM could not reconstruct a model response from the stream.",
        )

    completion = decode_completion(
        response,
        requested_model=requested_model,
        decision_format=decision_format,
        tool_data_formats=tool_data_formats,
    )
    if state.reasoning_text and completion.reasoning_content is None:
        completion.reasoning_content = state.reasoning_text
    return completion


@final
class _CompletionStreamState:
    def __init__(
        self,
        *,
        content_callback: Callable[[str], Coroutine[None, None, None]] | None,
        output_callback: OutputStreamCallback | None,
        reasoning_callback: Callable[[str], Coroutine[None, None, None]] | None,
        decision_format: StructuredDecisionFormat | None,
        tool_data_formats: dict[str, StructuredDataFormat],
    ) -> None:
        self._content_callback = content_callback
        self._output_callback = output_callback
        self._reasoning_callback = reasoning_callback
        self._decision_format = decision_format
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._structured_decoder = (
            JsonOutputStreamDecoder()
            if decision_format is not None and output_callback is not None
            else None
        )
        self._native_decoder = (
            NativeToolCallStreamDecoder(tool_data_formats)
            if tool_data_formats and output_callback is not None
            else None
        )

    @property
    def reasoning_text(self) -> str:
        return "".join(self._reasoning_parts)

    @property
    def content_text(self) -> str:
        return "".join(self._content_parts)

    async def feed(self, delta: _CompletionDelta | None) -> None:
        if delta is None:
            return

        reasoning = getattr(delta, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning:
            self._reasoning_parts.append(reasoning)
            if self._reasoning_callback is not None:
                await self._reasoning_callback(reasoning)

        content = delta.content
        if content:
            self._content_parts.append(content)
            if self._content_callback is not None:
                await self._content_callback(content)
            if self._structured_decoder is not None:
                assert self._decision_format is not None
                events: list[OutputStreamEvent] = []
                for event in self._structured_decoder.feed(content):
                    decoded = self._decision_format.decode_stream_event(event)
                    if decoded is not None:
                        events.append(decoded)
                await self._emit(events)

        if self._native_decoder is not None:
            await self._emit(self._native_decoder.feed(delta.tool_calls or []))

    async def finish(self) -> None:
        if self._native_decoder is not None:
            await self._emit(self._native_decoder.finish())

    async def _emit(self, events: list[OutputStreamEvent]) -> None:
        assert self._output_callback is not None
        for event in events:
            await self._output_callback(event)


__all__ = ["consume_completion_stream"]
