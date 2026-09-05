from __future__ import annotations

import json
from collections.abc import Sequence

from typing_extensions import final, override

from .._interfaces import InferenceStrategy
from .._tool_system import ToolRegistry
from ..event_system import EventPublisher
from ..exceptions import InvalidInferenceResponseError, UnknownToolDecisionError
from ..inference import FunctionInfo, HistoryItem, StepDecision
from ..streaming import ArgEvent, Scalar, StreamHandler, StringDelta, StringEnd
from . import events
from ._arg_stream import ToolArgStreamer
from ._client import LLMClient
from ._messages import LLMCompletion
from .exceptions import DecisionDecodingError, LLMCompletionDecodingError
from ._prompt_renderer import PromptRenderer, RejectedDecision
from ._tool_call_ids import ToolCallIdRegistry
from .result_format import ResultFormatFactory
from .step_decision import DecisionSpec
from .streaming import (
    OutputStreamEvent,
    StringDelta as OutputStringDelta,
    StringEnd as OutputStringEnd,
)
from .transports import (
    DecisionObserver,
    DecisionRequest,
    DecisionTransport,
)


@final
class _InvalidDecisionCompletionError(InvalidInferenceResponseError):
    def __init__(self, detail: str, completion: LLMCompletion) -> None:
        super().__init__(detail, raw_content=completion.content)
        self.completion = completion


@final
class _StrategyDecisionObserver(DecisionObserver):
    def __init__(
        self,
        publisher: EventPublisher,
        decision_spec: DecisionSpec,
        tool_arg_streamer: ToolArgStreamer | None,
    ) -> None:
        self._publisher = publisher
        self._decision_spec = decision_spec
        self._tool_arg_streamer = tool_arg_streamer

    @override
    async def before_request(self, prompt: str) -> None:
        await self._publisher.publish(
            events.BeforeLLMCall(prompt=prompt, decision_spec=self._decision_spec)
        )

    @override
    async def response_text(self, text: str) -> None:
        await self._publisher.publish(events.LLMTokenReceived(token=text))

    @override
    async def reasoning_text(self, text: str) -> None:
        await self._publisher.publish(events.LLMReasoningTokenReceived(token=text))

    @override
    async def output(self, event: OutputStreamEvent) -> None:
        streamer = self._tool_arg_streamer
        if streamer is None:
            return

        path = event.path
        if (
            len(path) == 3
            and path[0] == "tool_calls"
            and isinstance(path[1], int)
            and path[2] == "name"
            and isinstance(event, OutputStringEnd)
        ):
            streamer.identify_tool(path[1], event.value)
            return

        if (
            len(path) != 4
            or path[0] != "tool_calls"
            or not isinstance(path[1], int)
            or path[2] != "arguments"
            or not isinstance(path[3], str)
        ):
            return

        name = path[3]
        argument_event: ArgEvent
        if isinstance(event, OutputStringDelta):
            argument_event = StringDelta(name=name, text=event.text)
        elif isinstance(event, OutputStringEnd):
            argument_event = StringEnd(name=name, value=event.value)
        else:
            argument_event = Scalar(name=name, value=event.value)
        streamer.on_argument(path[1], argument_event)


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
        decision_spec = DecisionSpec.for_inference(
            output_type=function_info.return_type,
            tools=tools.get_all(),
            result_format_factory=self._result_format_factory,
        )
        rejected: RejectedDecision | None = None

        for attempt in range(self._max_repair_attempts + 1):
            request = DecisionRequest(
                function=function_info,
                decision_spec=decision_spec,
                history=tuple(history),
                rejected=rejected,
            )
            try:
                return await self._complete_once(
                    request,
                    tools,
                    publisher,
                )
            except InvalidInferenceResponseError as error:
                if attempt == self._max_repair_attempts:
                    raise
                await publisher.publish(
                    events.DecisionRepairAttempt(error=error, attempt=attempt + 1)
                )
                rejected = RejectedDecision(
                    content=(
                        _rejected_completion_content(error.completion)
                        if isinstance(error, _InvalidDecisionCompletionError)
                        else error.raw_content
                    ),
                    reason=error.detail,
                )

        raise AssertionError("unreachable")

    async def _complete_once(
        self,
        request: DecisionRequest,
        tools: ToolRegistry,
        publisher: EventPublisher,
    ) -> StepDecision:
        tool_call_ids = ToolCallIdRegistry()
        tool_arg_streamer = self._tool_arg_streamer(tools, tool_call_ids)
        observer = _StrategyDecisionObserver(
            publisher,
            request.decision_spec,
            tool_arg_streamer,
        )

        try:
            decoded = await self._decision_transport.request_decision(
                client=self.llm_client,
                prompt_renderer=self._prompt_renderer,
                request=request,
                observer=observer,
                stream=self._stream,
            )
        except (DecisionDecodingError, LLMCompletionDecodingError) as error:
            raise _InvalidDecisionCompletionError(
                f"LLM decision could not be decoded: {error}",
                error.completion,
            ) from error
        finally:
            if tool_arg_streamer is not None:
                await tool_arg_streamer.close()

        await publisher.publish(events.AfterLLMCall(decoded.completion))
        try:
            return request.decision_spec.validate(decoded.decision_data, tool_call_ids)
        except UnknownToolDecisionError as error:
            raise _InvalidDecisionCompletionError(
                f"LLM decision requested an unknown tool: {error.tool_name!r}",
                decoded.completion,
            ) from error
        except ValueError as error:
            raise _InvalidDecisionCompletionError(
                f"LLM decision failed validation: {error}",
                decoded.completion,
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


def _rejected_completion_content(completion: LLMCompletion) -> str | None:
    if not completion.tool_calls and completion.content is not None:
        return completion.content

    if not completion.tool_calls and completion.structured_output is None:
        return None

    response: dict[str, object] = {}
    if completion.content is not None:
        response["content"] = completion.content
    if completion.tool_calls:
        response["tool_calls"] = [
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments.tree,
            }
            for call in completion.tool_calls
        ]
    if completion.structured_output is not None:
        response["structured_output"] = completion.structured_output.tree
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))
