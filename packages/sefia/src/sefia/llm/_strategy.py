from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Never, Union

from pydantic import create_model

from .._interfaces import InferenceStrategy, ModelInspector
from ..event_system import EventPublisher
from ..exceptions import InvalidInferenceResponseError
from ..inference import (
    FinalAnswerDecision,
    FunctionInfo,
    HistoryItem,
    InferenceDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)
from . import events
from ._client import LLMClient
from ._messages import Message
from ._prompt_formatter import PromptFormatter

JsonDefault = Callable[[Any], Any]


@dataclass
class LLMToolCall:
    """A tool call requested by the inference strategy before an ID is assigned."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class _ExecutionDirector(ABC):
    """
    Abstract base class for directing the LLM's execution flow.
    Encapsulates the logic for building schemas, prompts, and processing
    decisions for a specific mode of operation.
    """

    def __init__(
        self,
        model_inspector: ModelInspector,
        output_type: Any,
        tools: list[dict],
    ):
        self.model_inspector = model_inspector
        self.output_type = output_type
        self.tools = tools

    @abstractmethod
    def build_decision_schema(self) -> dict:
        """Builds the JSON schema for the LLM's decision."""
        raise NotImplementedError

    @abstractmethod
    def build_system_prompt_addition(self, output_schema: dict) -> str:
        """Builds the core instruction part of the system prompt."""
        raise NotImplementedError

    @abstractmethod
    def process_decision(self, decision: _LLMDecision) -> InferenceDecision:
        """Processes the LLM's decision and returns an InferenceDecision."""
        raise NotImplementedError


class _ToolOnlyDirector(_ExecutionDirector):
    """Director for tool-only execution mode."""

    def build_decision_schema(self) -> dict:
        decision_model = create_model(
            "LLMDecision",
            tool_calls=(list[LLMToolCall], ...),
        )
        schema = self.model_inspector.get_schema_for_type(decision_model)
        schema["description"] = "The model for the LLM's decision on the next action."
        return schema

    def build_system_prompt_addition(self, output_schema: dict) -> str:
        core_instruction = (
            "Your task is to call tools. You MUST always populate the `tool_calls` "
            "field. There is no `final_answer` — you must never stop calling tools."
        )
        tool_definitions = [t.get("function", {}) for t in self.tools]
        return (
            f"\n\n### Response Instructions\n{core_instruction}\n"
            "\n### Available Tools\n"
            "Here is a list of tools you can call. Use their `name` in the `tool_calls` field.\n"
            f"{json.dumps(tool_definitions, indent=2, ensure_ascii=False)}\n"
            f"\n### Response Schema\n"
            f"Your response MUST be a single, valid, raw JSON object that strictly conforms to this JSON Schema:\n"
            f"{json.dumps(output_schema, ensure_ascii=False)}"
        )

    def process_decision(self, decision: _LLMDecision) -> InferenceDecision:
        if decision.tool_calls:
            return ToolCallDecision(
                calls=[
                    ToolCallRequest(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=tc.name,
                        arguments=tc.arguments,
                    )
                    for tc in decision.tool_calls
                ]
            )
        if decision.final_answer is not None:
            raise InvalidInferenceResponseError(
                "Return type is Never but LLM returned a final answer."
            )
        raise InvalidInferenceResponseError("LLM response must contain 'tool_calls'.")


class _ToolEnabledDirector(_ExecutionDirector):
    """Director for tool-enabled execution mode (tools or final answer)."""

    def build_decision_schema(self) -> dict:
        decision_model = create_model(
            "LLMDecision",
            final_answer=(Union[self.output_type, None], ...),
            tool_calls=(Union[list[LLMToolCall], None], ...),
        )
        schema = self.model_inspector.get_schema_for_type(decision_model)
        schema["description"] = "The model for the LLM's decision on the next action."
        return schema

    def build_system_prompt_addition(self, output_schema: dict) -> str:
        core_instruction = (
            "Your task is to decide the next step. You have two options:\n"
            "1. Call one or more tools by populating the `tool_calls` field.\n"
            "2. Provide the final answer by populating the `final_answer` field.\n\n"
            "You MUST populate both fields and set the unused field to null. "
            "Exactly one field must be non-null. "
            "Use `tool_calls` to gather more information, and use `final_answer` "
            "only when you have enough information to complete the entire task."
        )
        tool_definitions = [t.get("function", {}) for t in self.tools]
        return (
            f"\n\n### Response Instructions\n{core_instruction}\n"
            "\n### Available Tools\n"
            "Here is a list of tools you can call. Use their `name` in the `tool_calls` field.\n"
            f"{json.dumps(tool_definitions, indent=2, ensure_ascii=False)}\n"
            f"\n### Response Schema\n"
            f"Your response MUST be a single, valid, raw JSON object that strictly conforms to this JSON Schema:\n"
            f"{json.dumps(output_schema, ensure_ascii=False)}"
        )

    def process_decision(self, decision: _LLMDecision) -> InferenceDecision:
        if decision.tool_calls:
            return ToolCallDecision(
                calls=[
                    ToolCallRequest(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=tc.name,
                        arguments=tc.arguments,
                    )
                    for tc in decision.tool_calls
                ]
            )
        if decision.final_answer is not None:
            validated_answer = self.model_inspector.validate_and_create(
                self.output_type, decision.final_answer
            )
            return FinalAnswerDecision(answer=validated_answer)
        raise InvalidInferenceResponseError(
            "LLM response must contain either 'tool_calls' or a non-null 'final_answer'."
        )


class _OutputOnlyDirector(_ExecutionDirector):
    """Director for final-answer-only execution mode."""

    def build_decision_schema(self) -> dict:
        decision_model = create_model(
            "LLMDecision",
            final_answer=(self.output_type, ...),
        )
        schema = self.model_inspector.get_schema_for_type(decision_model)
        schema["description"] = "The model for the LLM's decision on the next action."
        return schema

    def build_system_prompt_addition(self, output_schema: dict) -> str:
        core_instruction = (
            "Your task is to provide a non-null final answer by populating the "
            "`final_answer` field. No tools are available. If the requested "
            "result is a collection and there are no results, return an empty "
            "collection instead of null."
        )
        return (
            f"\n\n### Response Instructions\n{core_instruction}\n"
            f"\n### Response Schema\n"
            f"Your response MUST be a single, valid, raw JSON object that strictly conforms to this JSON Schema:\n"
            f"{json.dumps(output_schema, ensure_ascii=False)}"
        )

    def process_decision(self, decision: _LLMDecision) -> InferenceDecision:
        if decision.final_answer is not None:
            validated_answer = self.model_inspector.validate_and_create(
                self.output_type, decision.final_answer
            )
            return FinalAnswerDecision(answer=validated_answer)
        raise InvalidInferenceResponseError(
            "LLM response must contain a non-null 'final_answer'."
        )


@dataclass
class _LLMDecision:
    """Typed stub for the dynamically created decision model."""

    final_answer: Any = None
    tool_calls: list[LLMToolCall] | None = None


class LLMInferenceStrategy(InferenceStrategy):
    """
    An inference strategy that uses an LLM to decide the next step.
    It unifies tool calls and final answers into a single structured output schema,
    making it compatible with a wide range of LLMs' JSON modes.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model_inspector: ModelInspector,
        prompt_formatter: PromptFormatter,
        json_default: JsonDefault | None = None,
        stream: bool = False,
    ):
        self.llm_client = llm_client
        self.model_inspector = model_inspector
        self._prompt_formatter = prompt_formatter
        self._json_default = json_default
        self._stream = stream

    def _create_director(
        self, output_type: Any, tools: list[dict]
    ) -> _ExecutionDirector:
        """Creates the appropriate execution director based on the context."""
        if output_type is Never:
            if not tools:
                raise ValueError(
                    "An @infer function returning Never must have tools available, "
                    "otherwise the inference loop can never make progress."
                )
            return _ToolOnlyDirector(self.model_inspector, output_type, tools)
        if tools:
            return _ToolEnabledDirector(self.model_inspector, output_type, tools)
        return _OutputOnlyDirector(self.model_inspector, output_type, tools)

    async def decide_next_step(
        self,
        function_info: FunctionInfo,
        history: list[HistoryItem],
        tools: list[dict],
        publisher: EventPublisher,
    ) -> InferenceDecision:
        director = self._create_director(function_info.return_type, tools)
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
        if self._stream:

            async def on_token(token: str):
                await publisher.publish(events.LLMTokenReceived(token=token))

            stream_callback = on_token

        response = await self.llm_client.complete(
            messages=messages,
            tools=None,
            output_schema=output_schema,
            stream_callback=stream_callback,
        )
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
            decision: _LLMDecision = self.model_inspector.validate_and_create(
                _LLMDecision, decision_data
            )
            return director.process_decision(decision)

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
            else "No arguments provided."
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
