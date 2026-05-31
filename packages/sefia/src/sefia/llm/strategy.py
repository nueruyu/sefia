import dataclasses
import json
import uuid
from typing import Any

from ..event_publisher import EventPublisher
from ..interfaces import InferenceStrategy, ModelInspector
from ..models import (
    FinalAnswerDecision,
    HistoryItem,
    InferenceDecision,
    LLMToolCall,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)
from . import events
from .client import LLMClient
from .messages import Message


def _pydantic_json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclasses.dataclass
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
        stream: bool = False,
    ):
        self.llm_client = llm_client
        self.model_inspector = model_inspector
        self._stream = stream

    def _build_llm_decision_schema(self, output_type: Any, tools: list[dict]) -> dict:
        """
        Dynamically creates a JSON Schema that represents the LLM's
        choice: either call tools or provide a final answer.
        """
        properties: dict[str, Any] = {
            "final_answer": {
                "oneOf": [
                    self.model_inspector.get_schema_for_type(output_type),
                    {"type": "null"},
                ]
            }
        }

        if tools:
            properties["tool_calls"] = {
                "oneOf": [
                    {
                        "type": "array",
                        "items": self.model_inspector.get_schema_for_type(LLMToolCall),
                    },
                    {"type": "null"},
                ]
            }

        return {
            "title": "LLMDecision",
            "description": "The model for the LLM's decision on the next action.",
            "type": "object",
            "properties": properties,
        }

    async def decide_next_step(
        self,
        instructions: str,
        arguments: dict[str, Any],
        history: list[HistoryItem],
        tools: list[dict],
        output_type: Any,
        publisher: EventPublisher,
    ) -> InferenceDecision:
        output_schema = self._build_llm_decision_schema(output_type, tools)

        messages = self._build_messages(
            instructions, arguments, history, output_schema, tools
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
            raise ValueError("LLM did not provide a response content.")

        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1]).strip()

            decision_data = json.loads(raw)
            decision = self.model_inspector.validate_and_create(
                _LLMDecision, decision_data
            )

            tool_calls: list[LLMToolCall] | None = getattr(decision, "tool_calls", None)
            if tool_calls:
                validated_calls = [
                    self.model_inspector.validate_and_create(LLMToolCall, tc)
                    for tc in tool_calls
                ]
                return ToolCallDecision(
                    calls=[
                        ToolCallRequest(
                            id=f"call_{uuid.uuid4().hex[:12]}",
                            name=tc.name,
                            arguments=tc.arguments,
                        )
                        for tc in validated_calls
                    ]
                )

            if decision.final_answer is not None:
                validated_answer = self.model_inspector.validate_and_create(
                    output_type, decision.final_answer
                )
                return FinalAnswerDecision(answer=validated_answer)

        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"LLM output failed validation against the master schema: {e}, content: {response.content}"
            ) from e

        raise ValueError(
            "LLM response must contain either 'tool_calls' or a non-null 'final_answer'."
        )

    def _build_messages(
        self,
        instructions: str,
        arguments: dict[str, Any],
        history: list[HistoryItem],
        output_schema: dict,
        tools: list[dict],
    ) -> list[Message]:
        messages: list[Message] = []

        system_content = instructions

        if tools:
            core_instruction = (
                "Your task is to decide the next step. You have two options:\n"
                "1. Call one or more tools by populating the `tool_calls` field.\n"
                "2. Provide the final answer by populating the `final_answer` field.\n\n"
                "You MUST use one, and only one, of these two fields in your response. "
                "Use `tool_calls` to gather more information, and use `final_answer` only when you have enough information to complete the entire task."
            )
        else:
            core_instruction = "Your task is to provide the final answer by populating the `final_answer` field. No tools are available."

        system_content += f"\n\n### Response Instructions\n{core_instruction}\n"

        if tools:
            tool_definitions = [t.get("function", {}) for t in tools]
            system_content += (
                "\n### Available Tools\n"
                "Here is a list of tools you can call. Use their `name` in the `tool_calls` field.\n"
                f"{json.dumps(tool_definitions, indent=2)}\n"
            )

        system_content += (
            f"\n### Response Schema\n"
            f"Your response MUST be a single, valid, raw JSON object that strictly conforms to this JSON Schema:\n"
            f"{json.dumps(output_schema)}"
        )
        messages.append(Message(role="system", content=system_content))

        arg_lines = "\n".join(
            f"- {name}: {value}" for name, value in arguments.items() if name != "self"
        )
        user_prompt = (
            f"Task arguments:\n{arg_lines}" if arg_lines else "No arguments provided."
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
                                    "arguments": json.dumps(call.arguments),
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
                        content=json.dumps(item.result, default=_pydantic_json_default),
                    )
                )

        return messages
