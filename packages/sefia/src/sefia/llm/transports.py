from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from typing_extensions import final, override

from ..streaming import ArgEvent, Scalar, StringDelta, StringEnd
from ._client import LLMClient
from ._messages import LLMResponse, Message
from ._prompted_response import PromptedJsonStreamExtractor, extract_prompted_json
from .llm_output import LLMOutput
from .step_decision import DecisionSpec
from .streaming import (
    JsonOutputStreamDecoder,
    OutputStreamEvent,
    StringDelta as OutputStringDelta,
    StringEnd as OutputStringEnd,
)


@dataclass(frozen=True)
class ResponseTextDelta:
    text: str


@dataclass(frozen=True)
class ReasoningTextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallIdentified:
    index: int
    name: str


@dataclass(frozen=True)
class ToolArgumentChanged:
    index: int
    event: ArgEvent


DecisionProgress = (
    ResponseTextDelta | ReasoningTextDelta | ToolCallIdentified | ToolArgumentChanged
)
DecisionProgressCallback = Callable[[DecisionProgress], Awaitable[None]]


@dataclass(frozen=True)
class DecisionResponse:
    output: LLMOutput
    raw: LLMResponse


class DecisionDecodingError(ValueError):
    def __init__(self, response: LLMResponse, reason: str) -> None:
        super().__init__(reason)
        self.response = response


class DecisionTransport(ABC):
    """Completes one decision through an LLM wire protocol."""

    @abstractmethod
    async def complete(
        self,
        client: LLMClient,
        prompt: str,
        decision: DecisionSpec,
        progress: DecisionProgressCallback | None,
    ) -> DecisionResponse: ...


class _ProgressReporter:
    def __init__(self, callback: DecisionProgressCallback) -> None:
        self._callback = callback

    async def response_text(self, text: str) -> None:
        await self._callback(ResponseTextDelta(text))

    async def reasoning_text(self, text: str) -> None:
        await self._callback(ReasoningTextDelta(text))

    async def output(self, event: OutputStreamEvent) -> None:
        converted = _convert_output_event(event)
        if converted is not None:
            await self._callback(converted)


@final
class StructuredDecisionTransport(DecisionTransport):
    @override
    async def complete(
        self,
        client: LLMClient,
        prompt: str,
        decision: DecisionSpec,
        progress: DecisionProgressCallback | None,
    ) -> DecisionResponse:
        reporter = _ProgressReporter(progress) if progress is not None else None

        response = await client.complete(
            messages=[Message(role="user", content=prompt)],
            tools=None,
            decision_model=decision,
            stream_callback=reporter.response_text if reporter is not None else None,
            output_callback=reporter.output if reporter is not None else None,
            reasoning_callback=reporter.reasoning_text
            if reporter is not None
            else None,
        )
        output = response.structured_output
        if output is None:
            raise DecisionDecodingError(
                response, "LLM client did not return structured output."
            )
        return DecisionResponse(output=output, raw=response)


@final
class PromptedDecisionTransport(DecisionTransport):
    @override
    async def complete(
        self,
        client: LLMClient,
        prompt: str,
        decision: DecisionSpec,
        progress: DecisionProgressCallback | None,
    ) -> DecisionResponse:
        reporter = _ProgressReporter(progress) if progress is not None else None
        stream_decoder = JsonOutputStreamDecoder() if reporter is not None else None
        extractor = PromptedJsonStreamExtractor() if reporter is not None else None

        async def on_text(text: str) -> None:
            assert reporter is not None
            assert stream_decoder is not None and extractor is not None
            await reporter.response_text(text)
            json_text = extractor.feed(text)
            if json_text:
                for event in stream_decoder.feed(json_text):
                    await reporter.output(event)

        response = await client.complete(
            messages=[Message(role="user", content=prompt)],
            tools=None,
            decision_model=None,
            stream_callback=on_text if reporter is not None else None,
            output_callback=None,
            reasoning_callback=reporter.reasoning_text
            if reporter is not None
            else None,
        )
        if response.content is None:
            raise DecisionDecodingError(
                response, "LLM did not provide response content."
            )
        try:
            output = LLMOutput.parse_json(extract_prompted_json(response.content))
        except ValueError as error:
            raise DecisionDecodingError(response, str(error)) from error
        return DecisionResponse(output=output, raw=response)


def _convert_output_event(event: OutputStreamEvent) -> DecisionProgress | None:
    path = event.path
    if len(path) == 3 and path[0] == "tool_calls" and isinstance(path[1], int):
        if path[2] == "name" and isinstance(event, OutputStringEnd):
            return ToolCallIdentified(index=path[1], name=event.value)
        return None
    if (
        len(path) == 4
        and path[0] == "tool_calls"
        and isinstance(path[1], int)
        and path[2] == "arguments"
        and isinstance(path[3], str)
    ):
        name = path[3]
        if isinstance(event, OutputStringDelta):
            argument_event: ArgEvent = StringDelta(name=name, text=event.text)
        elif isinstance(event, OutputStringEnd):
            argument_event = StringEnd(name=name, value=event.value)
        else:
            argument_event = Scalar(name=name, value=event.value)
        return ToolArgumentChanged(index=path[1], event=argument_event)
    return None


__all__ = [
    "DecisionProgress",
    "DecisionProgressCallback",
    "DecisionDecodingError",
    "DecisionResponse",
    "DecisionTransport",
    "PromptedDecisionTransport",
    "ReasoningTextDelta",
    "ResponseTextDelta",
    "StructuredDecisionTransport",
    "ToolArgumentChanged",
    "ToolCallIdentified",
]
