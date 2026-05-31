import json
import uuid
from typing import Any, Type, Union, cast

from pydantic import BaseModel, ValidationError, create_model

from ..event_publisher import EventPublisher
from ..interfaces import InferenceStrategy
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
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class _LLMDecision(BaseModel):
    """Typed stub for the dynamically created decision model."""

    final_answer: Any = None
    tool_calls: list[LLMToolCall] | None = None


class LLMInferenceStrategy(InferenceStrategy):
    """
    An inference strategy that uses an LLM to decide the next step.
    It unifies tool calls and final answers into a single structured output schema,
    making it compatible with a wide range of LLMs' JSON modes.
    """

    def __init__(self, llm_client: LLMClient, stream: bool = False):
        self.llm_client = llm_client
        self._stream = stream

    def _create_decision_model(
        self, output_type: Any, tools: list[dict]
    ) -> Type[_LLMDecision]:
        """
        Dynamically creates a Pydantic model that represents the LLM's
        choice: either call tools or provide a final answer.
        """
        fields: dict[str, Any] = {}
        fields["final_answer"] = (Union[output_type, None], None)

        if tools:
            fields["tool_calls"] = (Union[list[LLMToolCall], None], None)

        model_type = create_model(
            "LLMDecision",
            **fields,
            __doc__="The model for the LLM's decision on the next action.",
        )
        return cast(Type[_LLMDecision], model_type)

    async def decide_next_step(
        self,
        instructions: str,
        arguments: dict[str, Any],
        history: list[HistoryItem],
        tools: list[dict],
        output_type: Any,
        publisher: EventPublisher,
    ) -> InferenceDecision:
        decision_model = self._create_decision_model(output_type, tools)
        output_schema = decision_model.model_json_schema()

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
        await publisher.publish(events.AfterLLMCall(response=response))

        if response.content is None:
            raise ValueError("LLM did not provide a response content.")

        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1]).strip()

            decision = decision_model.model_validate_json(raw)

        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"LLM output failed validation against the master schema: {e}, content: {response.content}"
            ) from e

        tool_calls: list[LLMToolCall] | None = getattr(decision, "tool_calls", None)
        if tool_calls:
            return ToolCallDecision(
                calls=[
                    ToolCallRequest(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=tc.name,
                        arguments=tc.arguments,
                    )
                    for tc in tool_calls
                ]
            )

        if decision.final_answer is not None:
            return FinalAnswerDecision(answer=decision.final_answer)

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
