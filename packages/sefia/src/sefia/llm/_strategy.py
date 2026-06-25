from __future__ import annotations

import copy
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Never, Union

from pydantic import create_model

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


def _inline_local_refs(schema: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(schema)
    defs = schema.get("$defs", {})

    def resolve(value: Any, seen: frozenset[str] = frozenset()) -> Any:
        if isinstance(value, list):
            return [resolve(item, seen) for item in value]
        if not isinstance(value, dict):
            return value
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            key = ref.removeprefix("#/$defs/")
            if key not in defs or key in seen:
                return {k: resolve(v, seen) for k, v in value.items()}
            resolved = resolve(copy.deepcopy(defs[key]), seen | {key})
            if isinstance(resolved, dict):
                resolved.update({k: resolve(v, seen) for k, v in value.items() if k != "$ref"})
            return resolved
        return {k: resolve(v, seen) for k, v in value.items() if k != "$defs"}

    return resolve(schema)


def _tool_name(tool: dict[str, Any]) -> str | None:
    fn = tool.get("function")
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    return name if isinstance(name, str) and name else None


def _tool_arguments_schema(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    parameters = fn.get("parameters") if isinstance(fn, dict) else None
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}, "required": []}
    schema = _inline_local_refs(parameters)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    return schema


def _tool_call_item_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
    variants = []
    for tool in tools:
        name = _tool_name(tool)
        if name is None:
            continue
        variants.append(
            {
                "type": "object",
                "properties": {
                    "name": {"enum": [name]},
                    "arguments": _tool_arguments_schema(tool),
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )
    if not variants:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
    if len(variants) == 1:
        return variants[0]
    return {"anyOf": variants}


def _tool_calls_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "array", "minItems": 1, "items": _tool_call_item_schema(tools)}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> None:
    if "enum" in schema and value not in schema["enum"]:
        raise InvalidInferenceResponseError(f"LLM tool call value at {path} must be one of {schema['enum']!r}.")
    alternatives = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(alternatives, list):
        errors = []
        for alternative in alternatives:
            if not isinstance(alternative, dict):
                continue
            try:
                _validate_value(value, alternative, path)
                return
            except InvalidInferenceResponseError as e:
                errors.append(e)
        raise InvalidInferenceResponseError(f"LLM tool call value at {path} did not match schema: {errors[0] if errors else 'no alternatives matched'}")

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            raise InvalidInferenceResponseError(f"LLM tool call value at {path} must be an object.")
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                raise InvalidInferenceResponseError(f"LLM tool call value at {path} is missing required property {name!r}.")
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise InvalidInferenceResponseError(f"LLM tool call value at {path} has unknown properties: {sorted(unknown)!r}.")
        for name, subschema in properties.items():
            if name in value and isinstance(subschema, dict):
                _validate_value(value[name], subschema, f"{path}.{name}")
    elif schema_type == "array":
        if not isinstance(value, list):
            raise InvalidInferenceResponseError(f"LLM tool call value at {path} must be an array.")
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise InvalidInferenceResponseError(f"LLM tool call value at {path} must contain at least {min_items} item(s).")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_value(item, item_schema, f"{path}[{index}]")
    elif schema_type == "string":
        if not isinstance(value, str):
            raise InvalidInferenceResponseError(f"LLM tool call value at {path} must be a string.")
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            raise InvalidInferenceResponseError(f"LLM tool call value at {path} must contain at least {min_length} character(s).")
    elif schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidInferenceResponseError(f"LLM tool call value at {path} must be an integer.")
    elif schema_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InvalidInferenceResponseError(f"LLM tool call value at {path} must be a number.")
    elif schema_type == "boolean":
        if not isinstance(value, bool):
            raise InvalidInferenceResponseError(f"LLM tool call value at {path} must be a boolean.")
    elif schema_type == "null" and value is not None:
        raise InvalidInferenceResponseError(f"LLM tool call value at {path} must be null.")


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
        self._tool_argument_schemas = {
            name: _tool_arguments_schema(tool)
            for tool in tools
            if (name := _tool_name(tool)) is not None
        }

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

    def _build_tool_call_requests(self, tool_calls: list[LLMToolCall]) -> list[ToolCallRequest]:
        requests = []
        for tc in tool_calls:
            self._validate_tool_call(tc)
            requests.append(
                ToolCallRequest(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=tc.name,
                    arguments=tc.arguments,
                )
            )
        return requests

    def _validate_tool_call(self, tool_call: LLMToolCall) -> None:
        if not self._tool_argument_schemas:
            return
        arguments_schema = self._tool_argument_schemas.get(tool_call.name)
        if arguments_schema is None:
            raise InvalidInferenceResponseError(f"LLM requested unknown tool: {tool_call.name!r}.")
        _validate_value(tool_call.arguments, arguments_schema, f"{tool_call.name}.arguments")


class _ToolOnlyDirector(_ExecutionDirector):
    """Director for tool-only execution mode."""

    def build_decision_schema(self) -> dict:
        return {
            "type": "object",
            "description": "The model for the LLM's decision on the next action.",
            "properties": {"tool_calls": _tool_calls_schema(self.tools)},
            "required": ["tool_calls"],
            "additionalProperties": False,
        }

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
            return ToolCallDecision(calls=self._build_tool_call_requests(decision.tool_calls))
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
        schema = self.model_inspector.get_type_schema(decision_model)
        schema["description"] = "The model for the LLM's decision on the next action."
        schema.setdefault("properties", {})["tool_calls"] = _nullable(_tool_calls_schema(self.tools))
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
            return ToolCallDecision(calls=self._build_tool_call_requests(decision.tool_calls))
        if decision.final_answer is not None:
            validated_answer = self.model_inspector.validate(
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
        schema = self.model_inspector.get_type_schema(decision_model)
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
            validated_answer = self.model_inspector.validate(
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
        tools: ToolRegistry,
        publisher: EventPublisher,
    ) -> InferenceDecision:
        tool_schemas = [
            self.model_inspector.get_function_schema(tool.function, name=tool.name)
            for tool in tools.get_all()
        ]
        director = self._create_director(function_info.return_type, tool_schemas)
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
            decision: _LLMDecision = self.model_inspector.validate(
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


def _tool_stream_handlers(tools: ToolRegistry) -> dict[str, StreamHandler]:
    return {
        tool.name: tool.stream_handler
        for tool in tools.get_all()
        if tool.stream_handler is not None
    }
