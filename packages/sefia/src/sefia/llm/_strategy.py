from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, Never, Optional, Union

from pydantic import Field, create_model

from .._interfaces import InferenceStrategy, ModelInspector
from .._tool_system import ToolRegistry
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
from ..streaming import StreamHandler
from . import events
from ._arg_stream import ToolArgStreamer
from ._client import LLMClient
from ._messages import Message
from ._prompt_formatter import PromptFormatter

JsonDefault = Callable[[Any], Any]


@dataclass(frozen=True)
class _ToolSpec:
    """A tool prepared for the decision schema.

    ``schema`` is the human/LLM-facing tool description shown in the prompt, and
    ``arguments_model`` is a strict Pydantic model (from the ModelInspector) used
    to both constrain the response schema and validate the call arguments — no
    hand-written JSON schema or validator is involved.
    """

    name: str
    schema: dict
    arguments_model: type


@dataclass
class LLMToolCall:
    """A loosely-parsed tool call: a tool name plus a raw arguments mapping.

    Unknown tool names are accepted here so they can flow to the executor for a
    graceful "tool not found" response. Arguments of *known* tools are validated
    against their ``arguments_model`` while building the decision."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


def _tool_calls_type(tool_specs: list[_ToolSpec]) -> Any:
    """Build the strict ``tool_calls`` type used to generate the response schema:
    a non-empty list of per-tool call models, discriminated on ``name`` so the
    schema spells out each tool's own argument constraints."""
    call_models = [
        create_model(
            f"{spec.name}ToolCall",
            name=(Literal[spec.name], ...),
            arguments=(spec.arguments_model, ...),
        )
        for spec in tool_specs
    ]
    if len(call_models) == 1:
        item_type: Any = call_models[0]
    else:
        item_type = Annotated[Union[tuple(call_models)], Field(discriminator="name")]
    return Annotated[list[item_type], Field(min_length=1)]


class _ExecutionDirector(ABC):
    """
    Abstract base class for directing the LLM's execution flow.

    Each mode builds a single dynamic ``LLMDecision`` Pydantic model. The model
    inspector turns that model into the response schema and, later, validates the
    LLM's raw JSON against it — tool argument constraints included.
    """

    def __init__(
        self,
        model_inspector: ModelInspector,
        output_type: Any,
        tool_specs: list[_ToolSpec],
    ):
        self.model_inspector = model_inspector
        self.output_type = output_type
        self.tool_specs = tool_specs
        self._spec_by_name = {spec.name: spec for spec in tool_specs}
        # decision_model drives the response schema (strict, per-tool argument
        # constraints); parse_model validates the LLM's reply leniently so that
        # unknown tool names still reach the executor.
        self.decision_model = self._build_decision_model(strict=True)
        self.parse_model = self._build_decision_model(strict=False)

    @abstractmethod
    def _build_decision_model(self, strict: bool) -> type:
        """Builds the dynamic Pydantic model for the LLM's decision."""
        raise NotImplementedError

    def _tool_calls_field_type(self, strict: bool) -> Any:
        if strict:
            return _tool_calls_type(self.tool_specs)
        return list[LLMToolCall]

    def _required_fields(self) -> list[str] | None:
        """Fields the LLM must populate. The decision model parses leniently (a
        missing field is treated as null), but the response schema still asks the
        LLM for these fields. Return None to keep the model's own required set."""
        return None

    def build_decision_schema(self) -> dict:
        """Builds the JSON schema for the LLM's decision."""
        schema = dict(self.model_inspector.get_type_schema(self.decision_model))
        schema["description"] = "The model for the LLM's decision on the next action."
        required = self._required_fields()
        if required is not None:
            schema["required"] = required
        return schema

    @abstractmethod
    def build_system_prompt_addition(self, output_schema: dict) -> str:
        """Builds the core instruction part of the system prompt."""
        raise NotImplementedError

    @abstractmethod
    def process_decision(self, decision: Any) -> InferenceDecision:
        """Processes the validated decision and returns an InferenceDecision."""
        raise NotImplementedError

    def _tool_definitions(self) -> list[dict]:
        return [spec.schema.get("function", {}) for spec in self.tool_specs]

    def _tool_call_decision(self, tool_calls: list[LLMToolCall]) -> ToolCallDecision:
        calls = []
        for tc in tool_calls:
            spec = self._spec_by_name.get(tc.name)
            if spec is not None:
                # Validate a known tool's arguments against its model. Missing
                # required, empty, or unexpected arguments raise here and surface
                # as an invalid-response error.
                validated = self.model_inspector.validate(
                    spec.arguments_model, tc.arguments
                )
                arguments = validated.model_dump()
            else:
                # Unknown tool: let the executor report it as "tool not found".
                arguments = tc.arguments
            calls.append(
                ToolCallRequest(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=tc.name,
                    arguments=arguments,
                )
            )
        return ToolCallDecision(calls=calls)


_TOOL_DEFINITIONS_HEADER = (
    "\n### Available Tools\n"
    "Here is a list of tools you can call. Use their `name` in the `tool_calls` field.\n"
)
_RESPONSE_SCHEMA_HEADER = (
    "\n### Response Schema\n"
    "Your response MUST be a single, valid, raw JSON object that strictly "
    "conforms to this JSON Schema:\n"
)


class _ToolOnlyDirector(_ExecutionDirector):
    """Director for tool-only execution mode."""

    def _build_decision_model(self, strict: bool) -> type:
        return create_model(
            "LLMDecision",
            tool_calls=(Optional[self._tool_calls_field_type(strict)], None),
        )

    def _required_fields(self) -> list[str] | None:
        return ["tool_calls"]

    def build_system_prompt_addition(self, output_schema: dict) -> str:
        core_instruction = (
            "Your task is to call tools. You MUST always populate the `tool_calls` "
            "field. There is no `final_answer` — you must never stop calling tools."
        )
        return (
            f"\n\n### Response Instructions\n{core_instruction}\n"
            f"{_TOOL_DEFINITIONS_HEADER}"
            f"{json.dumps(self._tool_definitions(), indent=2, ensure_ascii=False)}\n"
            f"{_RESPONSE_SCHEMA_HEADER}"
            f"{json.dumps(output_schema, ensure_ascii=False)}"
        )

    def process_decision(self, decision: Any) -> InferenceDecision:
        if decision.tool_calls:
            return self._tool_call_decision(decision.tool_calls)
        raise InvalidInferenceResponseError("LLM response must contain 'tool_calls'.")


class _ToolEnabledDirector(_ExecutionDirector):
    """Director for tool-enabled execution mode (tools or final answer)."""

    def _build_decision_model(self, strict: bool) -> type:
        return create_model(
            "LLMDecision",
            final_answer=(Optional[self.output_type], None),
            tool_calls=(Optional[self._tool_calls_field_type(strict)], None),
        )

    def _required_fields(self) -> list[str] | None:
        return ["final_answer", "tool_calls"]

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
        return (
            f"\n\n### Response Instructions\n{core_instruction}\n"
            f"{_TOOL_DEFINITIONS_HEADER}"
            f"{json.dumps(self._tool_definitions(), indent=2, ensure_ascii=False)}\n"
            f"{_RESPONSE_SCHEMA_HEADER}"
            f"{json.dumps(output_schema, ensure_ascii=False)}"
        )

    def process_decision(self, decision: Any) -> InferenceDecision:
        if decision.tool_calls:
            return self._tool_call_decision(decision.tool_calls)
        if decision.final_answer is not None:
            return FinalAnswerDecision(answer=decision.final_answer)
        raise InvalidInferenceResponseError(
            "LLM response must contain either 'tool_calls' or a non-null 'final_answer'."
        )


class _OutputOnlyDirector(_ExecutionDirector):
    """Director for final-answer-only execution mode."""

    def _build_decision_model(self, strict: bool) -> type:
        return create_model(
            "LLMDecision",
            final_answer=(self.output_type, ...),
        )

    def build_system_prompt_addition(self, output_schema: dict) -> str:
        core_instruction = (
            "Your task is to provide a non-null final answer by populating the "
            "`final_answer` field. No tools are available. If the requested "
            "result is a collection and there are no results, return an empty "
            "collection instead of null."
        )
        return (
            f"\n\n### Response Instructions\n{core_instruction}\n"
            f"{_RESPONSE_SCHEMA_HEADER}"
            f"{json.dumps(output_schema, ensure_ascii=False)}"
        )

    def process_decision(self, decision: Any) -> InferenceDecision:
        if decision.final_answer is not None:
            return FinalAnswerDecision(answer=decision.final_answer)
        raise InvalidInferenceResponseError(
            "LLM response must contain a non-null 'final_answer'."
        )


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

    def _build_tool_specs(self, tools: ToolRegistry) -> list[_ToolSpec]:
        return [
            _ToolSpec(
                name=tool.name,
                schema=self.model_inspector.get_function_schema(
                    tool.function, name=tool.name
                ),
                arguments_model=self.model_inspector.get_arguments_model(
                    tool.function, name=tool.name
                ),
            )
            for tool in tools.get_all()
        ]

    def _create_director(
        self, output_type: Any, tool_specs: list[_ToolSpec]
    ) -> _ExecutionDirector:
        """Creates the appropriate execution director based on the context."""
        if output_type is Never:
            if not tool_specs:
                raise ValueError(
                    "An @infer function returning Never must have tools available, "
                    "otherwise the inference loop can never make progress."
                )
            return _ToolOnlyDirector(self.model_inspector, output_type, tool_specs)
        if tool_specs:
            return _ToolEnabledDirector(self.model_inspector, output_type, tool_specs)
        return _OutputOnlyDirector(self.model_inspector, output_type, tool_specs)

    async def decide_next_step(
        self,
        function_info: FunctionInfo,
        history: list[HistoryItem],
        tools: ToolRegistry,
        publisher: EventPublisher,
    ) -> InferenceDecision:
        tool_specs = self._build_tool_specs(tools)
        director = self._create_director(function_info.return_type, tool_specs)
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
            decision = self.model_inspector.validate(
                director.parse_model, decision_data
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


def _tool_stream_handlers(tools: ToolRegistry) -> dict[str, StreamHandler]:
    return {
        tool.name: tool.stream_handler
        for tool in tools.get_all()
        if tool.stream_handler is not None
    }
