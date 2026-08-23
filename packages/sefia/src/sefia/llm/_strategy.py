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
from ._decision_mode import LLMDecisionMode
from ._json_response import parse_json_response
from ._message_builder import (
    build_messages,
    build_native_messages,
    build_native_repair_messages,
    build_repair_messages,
)
from ._messages import Message
from ._native_tools import NativeToolSet
from ._prompt_formatter import PromptFormatter
from ._step_decision_prompt import (
    build_json_decision_prompt,
    build_native_tool_prompt,
    build_step_decision_prompt,
)
from ._tool_call_ids import ToolCallIdRegistry
from .step_decision import (
    StepDecisionMode,
    StepDecisionModel,
    StepDecisionSpec,
)
from .result_format import ResultFormatFactory
from .llm_output import LLMOutput
from .streaming import OutputStreamEvent

JsonDefault = Callable[[Any], Any]


@final
class LLMInferenceStrategy(InferenceStrategy):
    """Uses an LLM to decide the next inference step.

    Tool calls and results share one decision shape, generated through provider
    structured output or prompt-described JSON. Invalid responses may be retried
    with corrective feedback before ``InvalidInferenceResponseError`` propagates.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        result_format_factory: ResultFormatFactory,
        prompt_formatter: PromptFormatter,
        json_default: JsonDefault | None = None,
        stream: bool = False,
        max_repair_attempts: int = 2,
        decision_mode: LLMDecisionMode = LLMDecisionMode.STRUCTURED_OUTPUT,
    ):
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be non-negative")
        self.llm_client = llm_client
        self._result_format_factory = result_format_factory
        self._prompt_formatter = prompt_formatter
        self._json_default = json_default
        self._stream = stream
        self._max_repair_attempts = max_repair_attempts
        self._decision_mode = decision_mode

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
        decision_model = StepDecisionModel.from_spec(spec, self._result_format_factory)
        native_tools = (
            NativeToolSet.from_model(decision_model)
            if self._decision_mode is LLMDecisionMode.NATIVE_TOOLS and spec.tools
            else None
        )
        messages = self._build_messages(
            function_info, history, spec, decision_model, native_tools
        )

        attempt = 0
        while True:
            try:
                return await self._complete_once(
                    messages,
                    decision_model,
                    tools,
                    publisher,
                    native_tools,
                )
            except InvalidInferenceResponseError as error:
                if attempt >= self._max_repair_attempts:
                    raise
                attempt += 1
                await publisher.publish(
                    events.LLMResponseRepairAttempt(error=error, attempt=attempt)
                )
                repair_messages = (
                    build_native_repair_messages(error)
                    if native_tools is not None
                    else build_repair_messages(error)
                )
                messages = messages + repair_messages

    async def _complete_once(
        self,
        messages: list[Message],
        decision_model: StepDecisionModel,
        tools: ToolRegistry,
        publisher: EventPublisher,
        native_tools: NativeToolSet | None,
    ) -> StepDecision:
        await publisher.publish(
            events.BeforeLLMCall(
                messages=messages,
                tools=native_tools.definitions if native_tools is not None else None,
                decision_model=decision_model if native_tools is None else None,
            )
        )

        stream_callback = None
        reasoning_callback = None
        tool_stream_handlers = _tool_stream_handlers(tools)
        if (
            self._decision_mode in {LLMDecisionMode.JSON, LLMDecisionMode.NATIVE_TOOLS}
            and self._stream
            and tool_stream_handlers
        ):
            raise ValueError(
                f"{self._decision_mode.value} decision mode does not support "
                "streamed tool arguments."
            )
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

            async def on_output(event: OutputStreamEvent) -> None:
                if tool_arg_streamer is not None:
                    tool_arg_streamer.on_event(event)

            async def on_reasoning_token(token: str):
                await publisher.publish(events.LLMReasoningTokenReceived(token=token))

            stream_callback = on_token
            output_callback = on_output if tool_arg_streamer is not None else None
            reasoning_callback = on_reasoning_token
        else:
            output_callback = None

        try:
            response = await self.llm_client.complete(
                messages=messages,
                tools=native_tools.definitions if native_tools is not None else None,
                decision_model=(
                    decision_model
                    if self._decision_mode is LLMDecisionMode.STRUCTURED_OUTPUT
                    or (
                        self._decision_mode is LLMDecisionMode.NATIVE_TOOLS
                        and native_tools is None
                    )
                    else None
                ),
                stream_callback=stream_callback,
                output_callback=output_callback,
                reasoning_callback=reasoning_callback,
            )
        finally:
            if tool_arg_streamer is not None:
                await tool_arg_streamer.close()
        await publisher.publish(events.AfterLLMCall(response))

        if native_tools is not None:
            try:
                return native_tools.validate_calls(response.tool_calls, decision_model)
            except UnknownToolDecisionError as error:
                raise InvalidInferenceResponseError(
                    f"LLM output requested an unknown tool: {error.tool_name!r}",
                    raw_content=response.content,
                ) from error
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise InvalidInferenceResponseError(
                    f"LLM native tool output failed validation: {error}",
                    raw_content=response.content,
                ) from error

        if response.content is None and response.structured_output is None:
            raise InvalidInferenceResponseError(
                "LLM did not provide a response content."
            )

        try:
            decision_data = response.structured_output
            if decision_data is None:
                assert response.content is not None
                decision_data = parse_json_response(
                    response.content,
                    allow_surrounding_text=(
                        self._decision_mode is LLMDecisionMode.JSON
                    ),
                )
            if (
                self._decision_mode is LLMDecisionMode.JSON
                and decision_model.mode is StepDecisionMode.RESULT_ONLY
            ):
                decision_data = LLMOutput.from_object(
                    {
                        "decision": LLMOutput.from_scalar("result"),
                        "result": decision_data,
                    }
                )
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
        decision_model: StepDecisionModel | None = None,
        native_tools: NativeToolSet | None = None,
    ) -> list[Message]:
        decision_model = decision_model or StepDecisionModel.from_spec(
            spec, self._result_format_factory
        )
        if self._decision_mode is LLMDecisionMode.NATIVE_TOOLS and spec.tools:
            native_tools = native_tools or NativeToolSet.from_model(decision_model)
            return build_native_messages(
                function_info,
                history,
                build_native_tool_prompt(spec.mode, native_tools.result_tool_name),
                self._prompt_formatter,
                self._json_default,
            )
        decision_prompt = (
            build_step_decision_prompt(spec)
            if self._decision_mode
            in {LLMDecisionMode.STRUCTURED_OUTPUT, LLMDecisionMode.NATIVE_TOOLS}
            else build_json_decision_prompt(spec, decision_model)
        )
        return build_messages(
            function_info,
            history,
            decision_prompt,
            self._prompt_formatter,
            self._json_default,
            include_tool_call_ids=(
                self._decision_mode
                in {LLMDecisionMode.STRUCTURED_OUTPUT, LLMDecisionMode.NATIVE_TOOLS}
            ),
        )


def _tool_stream_handlers(tools: ToolRegistry) -> dict[str, StreamHandler]:
    return {
        tool.name: tool.stream_handler
        for tool in tools.get_all()
        if tool.stream_handler is not None
    }
