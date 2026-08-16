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
from ._message_builder import build_messages, build_repair_messages
from ._messages import Message
from ._prompt_formatter import PromptFormatter
from ._step_decision_prompt import build_step_decision_prompt
from ._tool_call_ids import ToolCallIdRegistry
from .json_schema import require_json_value
from .step_decision import (
    StepDecisionModel,
    StepDecisionSpec,
)
from .structured_output import StructuredValueSchemaFactory, to_structured_value
from .streaming import StructuredOutputEvent

JsonDefault = Callable[[Any], Any]


@final
class LLMInferenceStrategy(InferenceStrategy):
    """Uses an LLM to decide the next inference step.

    Tool calls and results share one structured output schema. Invalid
    structured responses may be retried with corrective feedback before
    ``InvalidInferenceResponseError`` propagates.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        structured_value_schema_factory: StructuredValueSchemaFactory,
        prompt_formatter: PromptFormatter,
        json_default: JsonDefault | None = None,
        stream: bool = False,
        max_repair_attempts: int = 2,
    ):
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be non-negative")
        self.llm_client = llm_client
        self._structured_value_schema_factory = structured_value_schema_factory
        self._prompt_formatter = prompt_formatter
        self._json_default = json_default
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
        spec = StepDecisionSpec.for_inference(
            name="StepDecision",
            output_type=function_info.return_type,
            tools=tools.get_all(),
        )
        decision_model = StepDecisionModel.from_spec(
            spec, self._structured_value_schema_factory
        )
        messages = self._build_messages(function_info, history, spec)

        attempt = 0
        while True:
            try:
                return await self._complete_once(
                    messages,
                    decision_model,
                    tools,
                    publisher,
                )
            except InvalidInferenceResponseError as error:
                if attempt >= self._max_repair_attempts:
                    raise
                attempt += 1
                await publisher.publish(
                    events.LLMResponseRepairAttempt(error=error, attempt=attempt)
                )
                messages = messages + build_repair_messages(error)

    async def _complete_once(
        self,
        messages: list[Message],
        decision_model: StepDecisionModel,
        tools: ToolRegistry,
        publisher: EventPublisher,
    ) -> StepDecision:
        await publisher.publish(
            events.BeforeLLMCall(
                messages=messages,
                tools=None,
                decision_model=decision_model,
            )
        )

        stream_callback = None
        reasoning_callback = None
        tool_stream_handlers = _tool_stream_handlers(tools)
        tool_arg_streamer = None
        tool_call_ids = ToolCallIdRegistry()
        if self._stream and tool_stream_handlers:
            tool_arg_streamer = ToolArgStreamer(
                tool_stream_handlers,
                tool_call_ids.get_or_create,
            )
        if self._stream:

            async def on_token(token: str):
                await publisher.publish(events.LLMTokenReceived(token=token))

            async def on_structured_output(event: StructuredOutputEvent) -> None:
                if tool_arg_streamer is not None:
                    tool_arg_streamer.on_event(event)

            async def on_reasoning_token(token: str):
                await publisher.publish(events.LLMReasoningTokenReceived(token=token))

            stream_callback = on_token
            structured_output_callback = (
                on_structured_output if tool_arg_streamer is not None else None
            )
            reasoning_callback = on_reasoning_token
        else:
            structured_output_callback = None

        try:
            response = await self.llm_client.complete(
                messages=messages,
                tools=None,
                decision_model=decision_model,
                stream_callback=stream_callback,
                structured_output_callback=structured_output_callback,
                reasoning_callback=reasoning_callback,
            )
        finally:
            if tool_arg_streamer is not None:
                await tool_arg_streamer.close()
        await publisher.publish(events.AfterLLMCall(response))

        if response.content is None and response.structured_output is None:
            raise InvalidInferenceResponseError(
                "LLM did not provide a response content."
            )

        try:
            decision_data = response.structured_output
            if decision_data is None:
                assert response.content is not None
                raw = response.content.strip()
                if raw.startswith("```"):
                    lines = raw.splitlines()
                    raw = "\n".join(lines[1:-1]).strip()
                decision_data = to_structured_value(require_json_value(json.loads(raw)))
            return decision_model.validate(decision_data, tool_call_ids)
        except UnknownToolDecisionError as error:
            raise InvalidInferenceResponseError(
                f"LLM output requested an unknown tool: {error.tool_name!r}",
                raw_content=response.content,
            ) from error
        except (json.JSONDecodeError, ValueError) as error:
            raise InvalidInferenceResponseError(
                f"LLM output failed validation against the master schema: {error}",
                raw_content=response.content,
            ) from error

    def _build_messages(
        self,
        function_info: FunctionInfo,
        history: Sequence[HistoryItem],
        spec: StepDecisionSpec,
    ) -> list[Message]:
        return build_messages(
            function_info,
            history,
            build_step_decision_prompt(spec),
            self._prompt_formatter,
            self._json_default,
        )


def _tool_stream_handlers(tools: ToolRegistry) -> dict[str, StreamHandler]:
    return {
        tool.name: tool.stream_handler
        for tool in tools.get_all()
        if tool.stream_handler is not None
    }
