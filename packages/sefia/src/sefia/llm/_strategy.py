from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import final, override

from .._interfaces import InferenceStrategy
from .._tool_system import ToolRegistry
from ..event_system import EventPublisher
from ..exceptions import InvalidInferenceResponseError, UnknownToolDecisionError
from ..inference import FunctionInfo, HistoryItem, StepDecision
from ..streaming import StreamHandler
from . import events
from ._arg_stream import ToolArgStreamer
from ._client import LLMClient
from ._prompt_renderer import DecisionPrompt, PromptRenderer, RejectedDecision
from ._tool_call_ids import ToolCallIdRegistry
from .result_format import ResultFormatFactory
from .step_decision import DecisionSpec
from .transports import (
    DecisionProgress,
    DecisionDecodingError,
    DecisionTransport,
    ReasoningTextDelta,
    ResponseTextDelta,
    ToolCallIdentified,
)


@final
class LLMInferenceStrategy(InferenceStrategy):
    """Uses an LLM to produce and validate the next inference decision."""

    def __init__(
        self,
        llm_client: LLMClient,
        result_format_factory: ResultFormatFactory,
        prompt_renderer: PromptRenderer,
        decision_transport: DecisionTransport,
        stream: bool = False,
        max_repair_attempts: int = 2,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be non-negative")
        self.llm_client = llm_client
        self._result_format_factory = result_format_factory
        self._prompt_renderer = prompt_renderer
        self._decision_transport = decision_transport
        self._stream = stream
        self._max_repair_attempts = max_repair_attempts

    @override
    async def decide_next_step(
        self,
        function_info: FunctionInfo,
        history: Sequence[HistoryItem],
        tools: ToolRegistry,
        publisher: EventPublisher,
    ) -> StepDecision:
        decision = DecisionSpec.for_inference(
            output_type=function_info.return_type,
            tools=tools.get_all(),
            result_format_factory=self._result_format_factory,
        )
        rejected: RejectedDecision | None = None

        for attempt in range(self._max_repair_attempts + 1):
            prompt = self._prompt_renderer.render(
                DecisionPrompt(
                    function=function_info,
                    decision=decision,
                    history=tuple(history),
                    rejected=rejected,
                )
            )
            try:
                return await self._complete_once(
                    prompt,
                    decision,
                    tools,
                    publisher,
                )
            except InvalidInferenceResponseError as error:
                if attempt == self._max_repair_attempts:
                    raise
                await publisher.publish(
                    events.LLMResponseRepairAttempt(error=error, attempt=attempt + 1)
                )
                rejected = RejectedDecision(
                    content=error.raw_content,
                    reason=error.detail,
                )

        raise AssertionError("unreachable")

    async def _complete_once(
        self,
        prompt: str,
        decision: DecisionSpec,
        tools: ToolRegistry,
        publisher: EventPublisher,
    ) -> StepDecision:
        await publisher.publish(events.BeforeLLMCall(prompt=prompt, decision=decision))

        tool_call_ids = ToolCallIdRegistry()
        tool_arg_streamer = self._tool_arg_streamer(tools, tool_call_ids)

        async def on_progress(progress: DecisionProgress) -> None:
            if isinstance(progress, ResponseTextDelta):
                await publisher.publish(events.LLMTokenReceived(token=progress.text))
            elif isinstance(progress, ReasoningTextDelta):
                await publisher.publish(
                    events.LLMReasoningTokenReceived(token=progress.text)
                )
            elif tool_arg_streamer is not None:
                if isinstance(progress, ToolCallIdentified):
                    tool_arg_streamer.identify_tool(progress.index, progress.name)
                else:
                    tool_arg_streamer.on_argument(progress.index, progress.event)

        try:
            response = await self._decision_transport.complete(
                client=self.llm_client,
                prompt=prompt,
                decision=decision,
                progress=on_progress if self._stream else None,
            )
        except DecisionDecodingError as error:
            raise InvalidInferenceResponseError(
                f"LLM output could not be decoded: {error}",
                raw_content=error.response.content,
            ) from error
        finally:
            if tool_arg_streamer is not None:
                await tool_arg_streamer.close()

        await publisher.publish(events.AfterLLMCall(response.raw))
        try:
            return decision.validate(response.output, tool_call_ids)
        except UnknownToolDecisionError as error:
            raise InvalidInferenceResponseError(
                f"LLM output requested an unknown tool: {error.tool_name!r}",
                raw_content=response.raw.content,
            ) from error
        except ValueError as error:
            raise InvalidInferenceResponseError(
                f"LLM output failed validation: {error}",
                raw_content=response.raw.content,
            ) from error

    def _tool_arg_streamer(
        self,
        tools: ToolRegistry,
        tool_call_ids: ToolCallIdRegistry,
    ) -> ToolArgStreamer | None:
        handlers = _tool_stream_handlers(tools)
        if not self._stream or not handlers:
            return None
        return ToolArgStreamer(handlers, tool_call_ids.get_or_create)


def _tool_stream_handlers(tools: ToolRegistry) -> dict[str, StreamHandler]:
    return {
        tool.name: tool.stream_handler
        for tool in tools.get_all()
        if tool.stream_handler is not None
    }
