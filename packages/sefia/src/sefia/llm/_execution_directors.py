import json
from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import final, override

from .._interfaces import (
    DecisionModel,
    DecisionModelBuilder,
    DecisionModelSpec,
    DecisionToolCall,
    LLMDecision,
    ResultLLMDecision,
    ToolCallsLLMDecision,
)
from .._tool_system import ToolEntry
from ..exceptions import InvalidInferenceResponseError
from ..inference import (
    InferenceDecision,
    ResultDecision,
    ToolCallDecision,
    ToolCallRequest,
)
from ._tool_call_ids import ToolCallIdRegistry
from .schema import LLMSchema

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
    "To call tools, select the `tool_calls` decision and provide each tool's "
    "name and arguments according to the response schema."
)
_RESULT_RESPONSE_FORMAT = (
    "To complete the task, select the `result` decision and provide the result "
    "according to the response schema."
)


class ExecutionDirector(ABC):
    """Directs one of the supported LLM execution modes."""

    def __init__(
        self,
        decision_builder: DecisionModelBuilder,
        output_type: Any,
        tools: list[ToolEntry],
    ):
        self.decision_builder = decision_builder
        self.output_type = output_type
        self.tools = tools
        self.decision_model = self._build_decision_model()

    @abstractmethod
    def _build_decision_model(self) -> DecisionModel:
        raise NotImplementedError

    @final
    def build_decision_schema(self) -> LLMSchema:
        return self.decision_model.schema()

    @abstractmethod
    def build_system_prompt_addition(self, output_schema: dict[str, Any]) -> str:
        raise NotImplementedError

    @final
    def process_response_data(
        self, data: Any, tool_call_ids: ToolCallIdRegistry | None = None
    ) -> InferenceDecision:
        decision = self.decision_model.validate(data)
        return self._process_decision(decision, tool_call_ids)

    @abstractmethod
    def _process_decision(
        self,
        decision: LLMDecision,
        tool_call_ids: ToolCallIdRegistry | None,
    ) -> InferenceDecision:
        raise NotImplementedError

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [tool.definition().to_dict() for tool in self.tools]

    def _tool_call_decision(
        self,
        tool_calls: list[DecisionToolCall],
        tool_call_ids: ToolCallIdRegistry,
    ) -> ToolCallDecision:
        return ToolCallDecision(
            calls=[
                ToolCallRequest(
                    id=tool_call_ids.get_or_create(index),
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                )
                for index, tool_call in enumerate(tool_calls)
            ]
        )


@final
class ToolOnlyDirector(ExecutionDirector):
    @override
    def _build_decision_model(self) -> DecisionModel:
        return self.decision_builder.build(
            DecisionModelSpec.tool_only(
                name="LLMDecision", output_type=self.output_type, tools=self.tools
            )
        )

    @override
    def build_system_prompt_addition(self, output_schema: dict[str, Any]) -> str:
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

    @override
    def _process_decision(
        self, decision: LLMDecision, tool_call_ids: ToolCallIdRegistry | None
    ) -> InferenceDecision:
        if isinstance(decision, ToolCallsLLMDecision):
            if tool_call_ids is None:
                raise RuntimeError("Tool call ids are required for a tool decision.")
            return self._tool_call_decision(decision.tool_calls, tool_call_ids)
        raise InvalidInferenceResponseError("LLM response must contain 'tool_calls'.")


@final
class ToolEnabledDirector(ExecutionDirector):
    @override
    def _build_decision_model(self) -> DecisionModel:
        return self.decision_builder.build(
            DecisionModelSpec.tool_enabled(
                name="LLMDecision", output_type=self.output_type, tools=self.tools
            )
        )

    @override
    def build_system_prompt_addition(self, output_schema: dict[str, Any]) -> str:
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

    @override
    def _process_decision(
        self, decision: LLMDecision, tool_call_ids: ToolCallIdRegistry | None
    ) -> InferenceDecision:
        if isinstance(decision, ToolCallsLLMDecision):
            if tool_call_ids is None:
                raise RuntimeError("Tool call ids are required for a tool decision.")
            return self._tool_call_decision(decision.tool_calls, tool_call_ids)
        return ResultDecision(result=decision.result)


@final
class OutputOnlyDirector(ExecutionDirector):
    @override
    def _build_decision_model(self) -> DecisionModel:
        return self.decision_builder.build(
            DecisionModelSpec.output_only(
                name="LLMDecision", output_type=self.output_type
            )
        )

    @override
    def build_system_prompt_addition(self, output_schema: dict[str, Any]) -> str:
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

    @override
    def _process_decision(
        self, decision: LLMDecision, tool_call_ids: ToolCallIdRegistry | None
    ) -> InferenceDecision:
        if isinstance(decision, ResultLLMDecision):
            return ResultDecision(result=decision.result)
        raise InvalidInferenceResponseError(
            "LLM response must contain a non-null 'result'."
        )
