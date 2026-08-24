from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Callable

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
from ._default_transports import EnvelopeToolCallTransport, StructuredResultTransport
from ._message_builder import build_initial_messages
from ._messages import Message
from ._prompt_formatter import PromptFormatter
from ._tool_call_ids import ToolCallIdRegistry
from .result_format import ResultFormatFactory
from .step_decision import StepDecisionModel, StepDecisionSpec
from .streaming import OutputStreamEvent
from .transports import ResultTransport, ToolCallTransport

JsonDefault = Callable[[Any], Any]


@final
class LLMInferenceStrategy(InferenceStrategy):
    """Uses an LLM to produce and validate the next inference decision."""

    def __init__(
        self,
        llm_client: LLMClient,
        result_format_factory: ResultFormatFactory,
        prompt_formatter: PromptFormatter,
        json_default: JsonDefault | None = None,
        stream: bool = False,
        max_repair_attempts: int = 2,
        tool_transport: ToolCallTransport | None = None,
        result_transport: ResultTransport | None = None,
    ):
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be non-negative")
        self.llm_client = llm_client
        self._result_format_factory = result_format_factory
        self._prompt_formatter = prompt_formatter
        self._json_default = json_default
        self._stream = stream
        self._max_repair_attempts = max_repair_attempts
        self._tool_transport = tool_transport or EnvelopeToolCallTransport()
        self._result_transport = result_transport or StructuredResultTransport()

    @override
    async def decide_next_step(
        self,
        function_info: FunctionInfo,
        history: Sequence[HistoryItem],
        tools: ToolRegistry,
        publisher: EventPublisher,
    ) -> StepDecision:
        spec = StepDecisionSpec.for_inference(
            name="StepDecision",
            output_type=function_info.return_type,
            tools=tools.get_all(),
        )
        model = StepDecisionModel.from_spec(spec, self._result_format_factory)
        tool_transport = self._tool_transport if spec.tools else None
        messages = self._build_messages(
            function_info, history, spec, model, tool_transport
        )

        attempt = 0
        while True:
            try:
                return await self._complete_once(
                    messages, model, tools, publisher, tool_transport
                )
            except InvalidInferenceResponseError as error:
                if attempt >= self._max_repair_attempts:
                    raise
                attempt += 1
                await publisher.publish(
                    events.LLMResponseRepairAttempt(error=error, attempt=attempt)
                )
                transport = tool_transport or self._result_transport
                messages = messages + transport.repair_messages(error)

    async def _complete_once(
        self,
        messages: list[Message],
        model: StepDecisionModel,
        tools: ToolRegistry,
        publisher: EventPublisher,
        tool_transport: ToolCallTransport | None,
    ) -> StepDecision:
        definitions = (
            list(tool_transport.definitions(model) or ()) if tool_transport else []
        )
        definitions.extend(self._result_transport.definitions(model) or ())
        request_definitions = definitions or None
        decision_model = (
            tool_transport.decision_model(model) if tool_transport else None
        ) or self._result_transport.decision_model(model)
        await publisher.publish(
            events.BeforeLLMCall(
                messages=messages,
                tools=request_definitions,
                decision_model=decision_model,
            )
        )

        tool_stream_handlers = _tool_stream_handlers(tools)
        if (
            tool_transport is not None
            and not tool_transport.supports_arg_streaming
            and self._stream
            and tool_stream_handlers
        ):
            raise ValueError(
                "The selected tool transport does not support streamed tool arguments."
            )

        stream_callback = None
        output_callback = None
        reasoning_callback = None
        tool_arg_streamer = None
        tool_call_ids = ToolCallIdRegistry()
        if self._stream and tool_stream_handlers:
            tool_arg_streamer = ToolArgStreamer(
                tool_stream_handlers, tool_call_ids.get_or_create
            )
        if self._stream:

            async def on_token(token: str) -> None:
                await publisher.publish(events.LLMTokenReceived(token=token))

            async def on_output(event: OutputStreamEvent) -> None:
                if tool_arg_streamer is not None:
                    tool_arg_streamer.on_event(event)

            async def on_reasoning_token(token: str) -> None:
                await publisher.publish(events.LLMReasoningTokenReceived(token=token))

            stream_callback = on_token
            output_callback = on_output if tool_arg_streamer is not None else None
            reasoning_callback = on_reasoning_token

        try:
            response = await self.llm_client.complete(
                messages=messages,
                tools=request_definitions,
                decision_model=decision_model,
                stream_callback=stream_callback,
                output_callback=output_callback,
                reasoning_callback=reasoning_callback,
            )
        finally:
            if tool_arg_streamer is not None:
                await tool_arg_streamer.close()
        await publisher.publish(events.AfterLLMCall(response))

        try:
            if tool_transport is not None:
                tool_decision = tool_transport.decode(response, model, tool_call_ids)
                if tool_decision is not None:
                    return tool_decision
            result_decision = self._result_transport.decode(response, model)
            if result_decision is not None:
                return result_decision
            raise ValueError("The response contains no allowed decision.")
        except UnknownToolDecisionError as error:
            raise InvalidInferenceResponseError(
                f"LLM output requested an unknown tool: {error.tool_name!r}",
                raw_content=response.content,
            ) from error
        except (json.JSONDecodeError, ValueError) as error:
            raise InvalidInferenceResponseError(
                f"LLM output failed validation: {error}",
                raw_content=response.content,
            ) from error

    def _build_messages(
        self,
        function_info: FunctionInfo,
        history: Sequence[HistoryItem],
        spec: StepDecisionSpec,
        model: StepDecisionModel,
        tool_transport: ToolCallTransport | None,
    ) -> list[Message]:
        prompt = self._result_transport.prompt(spec, model)
        if tool_transport is not None:
            prompt = tool_transport.prompt(spec, model) + prompt
        messages = build_initial_messages(function_info, prompt, self._prompt_formatter)
        history_transport = tool_transport or self._tool_transport
        if history:
            messages.extend(
                history_transport.render_history(history, self._json_default)
            )
        return messages


def _tool_stream_handlers(tools: ToolRegistry) -> dict[str, StreamHandler]:
    return {
        tool.name: tool.stream_handler
        for tool in tools.get_all()
        if tool.stream_handler is not None
    }
