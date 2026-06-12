import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Union

from pydantic import create_model

from .._interfaces import InferenceStrategy, ModelInspector, PromptFormatter
from ..event_system import EventPublisher
from ..inference import (
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

JsonDefault = Callable[[Any], Any]


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

    def _build_llm_decision_schema(self, output_type: Any, tools: list[dict]) -> dict:
        """
        Dynamically creates a JSON Schema that represents the LLM's
        choice: either call tools or provide a final answer.
        """
        if tools:
            decision_model = create_model(
                "LLMDecision",
                final_answer=(Union[output_type, None], ...),
                tool_calls=(Union[list[LLMToolCall], None], ...),
            )
        else:
            decision_model = create_model(
                "LLMDecision",
                final_answer=(output_type, ...),
            )

        result = self.model_inspector.get_schema_for_type(decision_model)
        result["description"] = "The model for the LLM's decision on the next action."
        return result

    async def decide_next_step(
        self,
        instructions: str,
        arguments: dict[str, Any],
        argument_type_hints: dict[str, Any],
        history: list[HistoryItem],
        tools: list[dict],
        output_type: Any,
        publisher: EventPublisher,
    ) -> InferenceDecision:
        output_schema = self._build_llm_decision_schema(output_type, tools)

        messages = self._build_messages(
            instructions,
            arguments,
            argument_type_hints,
            history,
            output_schema,
            tools,
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
        argument_type_hints: dict[str, Any],
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
                "You MUST populate both fields and set the unused field to null. "
                "Exactly one field must be non-null. "
                "Use `tool_calls` to gather more information, and use `final_answer` only when you have enough information to complete the entire task."
            )
        else:
            core_instruction = (
                "Your task is to provide a non-null final answer by populating the "
                "`final_answer` field. No tools are available. If the requested "
                "result is a collection and there are no results, return an empty "
                "collection instead of null."
            )

        system_content += f"\n\n### Response Instructions\n{core_instruction}\n"

        if tools:
            tool_definitions = [t.get("function", {}) for t in tools]
            system_content += (
                "\n### Available Tools\n"
                "Here is a list of tools you can call. Use their `name` in the `tool_calls` field.\n"
                f"{json.dumps(tool_definitions, indent=2, ensure_ascii=False)}\n"
            )

        system_content += (
            f"\n### Response Schema\n"
            f"Your response MUST be a single, valid, raw JSON object that strictly conforms to this JSON Schema:\n"
            f"{json.dumps(output_schema, ensure_ascii=False)}"
        )
        messages.append(Message(role="system", content=system_content))

        prompt_arguments = {
            name: value for name, value in arguments.items() if name != "self"
        }
        user_prompt = (
            "Task arguments are XML. Values in <string> may be wrapped in "
            "CDATA and should be read as raw text.\n\n"
            f"{self._prompt_formatter.format_arguments(prompt_arguments, argument_type_hints)}"
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
