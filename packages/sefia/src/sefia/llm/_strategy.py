from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Never

from .._interfaces import (
    DecisionModel,
    DecisionModelSpec,
    DecisionToolCall,
    InferenceStrategy,
    LLMDecision,
    ModelBackend,
    ResultLLMDecision,
    ToolCallsLLMDecision,
)
from .._tool_system import Tool, ToolRegistry
from ..event_system import EventPublisher
from ..exceptions import InvalidInferenceResponseError, UnknownToolDecisionError
from ..inference import (
    FunctionInfo,
    HistoryItem,
    InferenceDecision,
    ResultDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)
from ..streaming import StreamHandler
from . import events
from ._arg_stream import ToolArgStreamer
from ._client import LLMClient
from ._messages import Message
from ._prompt_formatter import PromptFormatter

JsonDefault = Callable[[Any], Any]


class _ExecutionDirector(ABC):
    """
    Abstract base class for directing the LLM's execution flow.
    """

    def __init__(
        self,
        model_backend: ModelBackend,
        output_type: Any,
        tools: list[Tool],
    ):
        self.model_backend = model_backend
        self.output_type = output_type
        self.tools = tools
        self.decision_model = self._build_decision_model()

    @abstractmethod
    def _build_decision_model(self) -> DecisionModel:
        """Builds the model for the LLM's decision."""
        raise NotImplementedError

    def build_decision_schema(self) -> dict:
        """Builds the JSON schema for the LLM's decision."""
        return self.decision_model.schema()

    @abstractmethod
    def build_system_prompt_addition(self, output_schema: dict) -> str:
        """Builds the core instruction part of the system prompt."""
        raise NotImplementedError

    def process_response_data(self, data: Any) -> InferenceDecision:
        """Validate raw decision data and convert it to an inference decision."""
        return self._process_decision(self.decision_model.validate(data))

    @abstractmethod
    def _process_decision(self, decision: LLMDecision) -> InferenceDecision:
        """Convert a validated decision to an inference decision."""
        raise NotImplementedError

    def _tool_definitions(self) -> list[dict]:
        return [
            self.model_backend.get_function_schema(
                tool.schema,
                name=tool.name,
            ).get("function", {})
            for tool in self.tools
        ]

    def _tool_call_decision(
        self, tool_calls: list[DecisionToolCall]
    ) -> ToolCallDecision:
        calls = []
        for tc in tool_calls:
            calls.append(
                ToolCallRequest(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=tc.name,
                    arguments=tc.arguments,
                )
            )
        return ToolCallDecision(calls=calls)


_TOOL_DEFINITIONS_HEADER = (
    "\n### Available Tools\n"
    "Here is a list of tools you can call. Use their `name` in the `tool_calls` field.\n"
)
_RESPONSE_FORMAT_HEADER = (
    "\n### Response Format\n"
    "Your response MUST be a single, valid, raw JSON object. Do not include "
    "prose, markdown, or code fences.\n"
)
_TOOL_CALLS_RESPONSE_FORMAT = (
    'Use this shape to call tools: {"decision":"tool_calls",'
    '"tool_calls":[{"name":"<tool name>","arguments":{...}}]}.'
)
_RESULT_RESPONSE_FORMAT = (
    'Use this shape to complete the task: {"decision":"result","result":...}.'
)


class _ToolOnlyDirector(_ExecutionDirector):
    """Director for tool-only execution mode."""

    def _build_decision_model(self) -> DecisionModel:
        return self.model_backend.build_decision_model(
            DecisionModelSpec.tool_only(
                name="LLMDecision",
                output_type=self.output_type,
                tools=self.tools,
            )
        )

    def build_system_prompt_addition(self, output_schema: dict) -> str:
        core_instruction = (
            "Your task is to call tools. You MUST set `decision` to `tool_calls` "
            "and populate the `tool_calls` field. There is no `result` — "
            "you must never stop calling tools."
        )
        return (
            f"\n\n### Response Instructions\n{core_instruction}\n"
            f"{_TOOL_DEFINITIONS_HEADER}"
            f"{json.dumps(self._tool_definitions(), indent=2, ensure_ascii=False)}\n"
            f"{_RESPONSE_FORMAT_HEADER}"
            f"{_TOOL_CALLS_RESPONSE_FORMAT}"
        )

    def _process_decision(self, decision: LLMDecision) -> InferenceDecision:
        if isinstance(decision, ToolCallsLLMDecision):
            return self._tool_call_decision(decision.tool_calls)
        raise InvalidInferenceResponseError("LLM response must contain 'tool_calls'.")


class _ToolEnabledDirector(_ExecutionDirector):
    """Director for tool-enabled execution mode (tools or result)."""

    def _build_decision_model(self) -> DecisionModel:
        return self.model_backend.build_decision_model(
            DecisionModelSpec.tool_enabled(
                name="LLMDecision",
                output_type=self.output_type,
                tools=self.tools,
            )
        )

    def build_system_prompt_addition(self, output_schema: dict) -> str:
        core_instruction = (
            "Your task is to decide the next step. You have two options:\n"
            "1. Call one or more tools by setting `decision` to `tool_calls` "
            "and populating the `tool_calls` field.\n"
            "2. Complete the task by setting `decision` to `result` "
            "and populating the `result` field.\n\n"
            "Use `tool_calls` to gather more information, and use `result` "
            "only when you have enough information to complete the entire task."
        )
        return (
            f"\n\n### Response Instructions\n{core_instruction}\n"
            f"{_TOOL_DEFINITIONS_HEADER}"
            f"{json.dumps(self._tool_definitions(), indent=2, ensure_ascii=False)}\n"
            f"{_RESPONSE_FORMAT_HEADER}"
            f"{_TOOL_CALLS_RESPONSE_FORMAT}\n"
            f"{_RESULT_RESPONSE_FORMAT}"
        )

    def _process_decision(self, decision: LLMDecision) -> InferenceDecision:
        if isinstance(decision, ToolCallsLLMDecision):
            return self._tool_call_decision(decision.tool_calls)
        if isinstance(decision, ResultLLMDecision):
            return ResultDecision(result=decision.result)


class _OutputOnlyDirector(_ExecutionDirector):
    """Director for result-only execution mode."""

    def _build_decision_model(self) -> DecisionModel:
        return self.model_backend.build_decision_model(
            DecisionModelSpec.output_only(
                name="LLMDecision",
                output_type=self.output_type,
            )
        )

    def build_system_prompt_addition(self, output_schema: dict) -> str:
        core_instruction = (
            "Your task is to provide a non-null result by setting `decision` "
            "to `result` and populating the `result` field. No tools are "
            "available. If the requested result is a collection and there are "
            "no results, return an empty collection instead of null."
        )
        return (
            f"\n\n### Response Instructions\n{core_instruction}\n"
            f"{_RESPONSE_FORMAT_HEADER}"
            f"{_RESULT_RESPONSE_FORMAT}"
        )

    def _process_decision(self, decision: LLMDecision) -> InferenceDecision:
        if isinstance(decision, ResultLLMDecision):
            return ResultDecision(result=decision.result)
        raise InvalidInferenceResponseError(
            "LLM response must contain a non-null 'result'."
        )


class LLMInferenceStrategy(InferenceStrategy):
    """
    An inference strategy that uses an LLM to decide the next step.
    It unifies tool calls and results into a single structured output schema,
    making it compatible with a wide range of LLMs' JSON modes.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model_backend: ModelBackend,
        prompt_formatter: PromptFormatter,
        json_default: JsonDefault | None = None,
        stream: bool = False,
    ):
        self.llm_client = llm_client
        self.model_backend = model_backend
        self._prompt_formatter = prompt_formatter
        self._json_default = json_default
        self._stream = stream

    def _create_director(
        self, output_type: Any, tools: list[Tool]
    ) -> _ExecutionDirector:
        """Creates the appropriate execution director based on the context."""
        if output_type is Never:
            if not tools:
                raise ValueError(
                    "An @infer function returning Never must have tools available, "
                    "otherwise the inference loop can never make progress."
                )
            return _ToolOnlyDirector(self.model_backend, output_type, tools)
        if tools:
            return _ToolEnabledDirector(self.model_backend, output_type, tools)
        return _OutputOnlyDirector(self.model_backend, output_type, tools)

    async def decide_next_step(
        self,
        function_info: FunctionInfo,
        history: list[HistoryItem],
        tools: ToolRegistry,
        publisher: EventPublisher,
    ) -> InferenceDecision:
        director = self._create_director(function_info.return_type, tools.get_all())
        output_schema = director.build_decision_schema()
        messages = self._build_messages(
            function_info,
            history,
            output_schema,
            director,
        )

        await publisher.publish(
            events.BeforeLLMCall(
                messages=messages,
                tools=None,
                output_schema=output_schema,
            )
        )

        stream_callback = None
        tool_stream_handlers = _tool_stream_handlers(tools)
        tool_arg_streamer = None
        if self._stream and tool_stream_handlers:
            tool_arg_streamer = ToolArgStreamer(tool_stream_handlers)
        if self._stream:

            async def on_token(token: str):
                if tool_arg_streamer is not None:
                    tool_arg_streamer.on_token(token)
                await publisher.publish(events.LLMTokenReceived(token=token))

            stream_callback = on_token

        try:
            response = await self.llm_client.complete(
                messages=messages,
                tools=None,
                output_schema=output_schema,
                stream_callback=stream_callback,
            )
        finally:
            if tool_arg_streamer is not None:
                await tool_arg_streamer.close()
        await publisher.publish(events.AfterLLMCall(response))

        if response.content is None:
            raise InvalidInferenceResponseError(
                "LLM did not provide a response content."
            )

        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1]).strip()

            decision_data = json.loads(raw)
            return director.process_response_data(decision_data)

        except UnknownToolDecisionError as e:
            raise InvalidInferenceResponseError(
                f"LLM output requested an unknown tool: {e.tool_name!r}, content: {response.content}"
            ) from e
        except (json.JSONDecodeError, ValueError) as e:
            raise InvalidInferenceResponseError(
                f"LLM output failed validation against the master schema: {e}, content: {response.content}"
            ) from e

    def _build_messages(
        self,
        function_info: FunctionInfo,
        history: list[HistoryItem],
        output_schema: dict,
        director: _ExecutionDirector,
    ) -> list[Message]:
        messages: list[Message] = []

        system_prompt_addition = director.build_system_prompt_addition(output_schema)
        system_content = function_info.instructions + system_prompt_addition
        messages.append(Message(role="system", content=system_content))

        prompt_arguments = {
            name: value
            for name, value in function_info.bound_arguments.items()
            if name != "self"
        }
        user_prompt = (
            "Task arguments are XML. Values in <string> may be wrapped in "
            "CDATA and should be read as raw text.\n\n"
            f"{self._prompt_formatter.format_arguments(prompt_arguments, function_info.type_hints)}"
            if prompt_arguments
            else (
                "This inference call has no direct function arguments. "
                "Follow the system instructions and use the conversation/tool "
                "history for any available context."
            )
        )
        messages.append(Message(role="user", content=user_prompt))

        for item in history:
            if isinstance(item, ToolCallDecision):
                messages.append(
                    Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(
                                        call.arguments, ensure_ascii=False
                                    ),
                                },
                            }
                            for call in item.calls
                        ],
                    )
                )
            elif isinstance(item, ToolCallResult):
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=item.tool_call_id,
                        content=json.dumps(
                            item.result,
                            default=self._json_default,
                            ensure_ascii=False,
                        ),
                    )
                )

        return messages


def _tool_stream_handlers(tools: ToolRegistry) -> dict[str, StreamHandler]:
    return {
        tool.name: tool.stream_handler
        for tool in tools.get_all()
        if tool.stream_handler is not None
    }
