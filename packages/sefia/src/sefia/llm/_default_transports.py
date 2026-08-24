from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import final, override

from ..exceptions import InvalidInferenceResponseError
from ..inference import HistoryItem, ResultDecision, ToolCallsDecision
from ._json_response import parse_json_response
from ._message_builder import render_envelope_history, render_prompt_json_history
from ._messages import LLMResponse, Message
from ._step_decision_prompt import (
    build_json_decision_prompt,
    build_step_decision_prompt,
)
from ._tool_call_ids import ToolCallIdRegistry
from .llm_output import LLMOutput
from .step_decision import StepDecisionMode, StepDecisionModel, StepDecisionSpec
from .transports import JsonDefault, ResultTransport, ToolCallTransport, ToolDefinition


def _repair_messages(error: InvalidInferenceResponseError) -> list[Message]:
    messages: list[Message] = []
    if error.raw_content:
        messages.append(Message(role="assistant", content=error.raw_content))
        content_note = ""
    else:
        content_note = "Your previous response was empty.\n"
    messages.append(
        Message(
            role="user",
            content=(
                "Your previous response was invalid and could not be used as the "
                "required decision JSON.\n"
                f"Error: {error.detail}\n{content_note}"
                "Respond again with exactly one valid raw JSON value matching the "
                "response contract. Do not include prose, markdown, or code fences."
            ),
        )
    )
    return messages


def _output(response: LLMResponse, *, surrounding_text: bool) -> LLMOutput:
    if response.structured_output is not None:
        return response.structured_output
    if response.content is None:
        raise ValueError("LLM did not provide a response content.")
    return parse_json_response(
        response.content, allow_surrounding_text=surrounding_text
    )


def _decision_name(value: LLMOutput) -> str | None:
    if not isinstance(value.data, dict):
        return None
    decision = value.data.get("decision")
    return decision if isinstance(decision, str) else None


class EnvelopeToolCallTransport(ToolCallTransport):
    @property
    @override
    def supports_arg_streaming(self) -> bool:
        return True

    @override
    def definitions(self, model: StepDecisionModel) -> list[ToolDefinition] | None:
        del model
        return None

    @override
    def decision_model(self, model: StepDecisionModel) -> StepDecisionModel | None:
        return model

    @override
    def prompt(self, spec: StepDecisionSpec, model: StepDecisionModel) -> str:
        del model
        return build_step_decision_prompt(spec) if spec.tools else ""

    @override
    def render_history(
        self, history: Sequence[HistoryItem], json_default: JsonDefault | None
    ) -> list[Message]:
        return render_envelope_history(history, json_default)

    @override
    def decode(
        self,
        response: LLMResponse,
        model: StepDecisionModel,
        tool_call_ids: ToolCallIdRegistry,
    ) -> ToolCallsDecision | None:
        output = _output(response, surrounding_text=False)
        if _decision_name(output) != "tool_calls":
            return None
        decision = model.validate(output, tool_call_ids)
        assert isinstance(decision, ToolCallsDecision)
        return decision

    @override
    def repair_messages(self, error: InvalidInferenceResponseError) -> list[Message]:
        return _repair_messages(error)


@final
class StructuredResultTransport(ResultTransport):
    @override
    def definitions(self, model: StepDecisionModel) -> list[ToolDefinition] | None:
        del model
        return None

    @override
    def decision_model(self, model: StepDecisionModel) -> StepDecisionModel | None:
        return model.result_only() if model.result is not None else None

    @override
    def prompt(self, spec: StepDecisionSpec, model: StepDecisionModel) -> str:
        del model
        return build_step_decision_prompt(spec) if not spec.tools else ""

    @override
    def decode(
        self, response: LLMResponse, model: StepDecisionModel
    ) -> ResultDecision | None:
        output = _output(response, surrounding_text=False)
        if _decision_name(output) != "result":
            return None
        decision = model.validate(output, None)
        assert isinstance(decision, ResultDecision)
        return decision

    @override
    def repair_messages(self, error: InvalidInferenceResponseError) -> list[Message]:
        return _repair_messages(error)


@final
class PromptJsonToolCallTransport(EnvelopeToolCallTransport):
    @property
    @override
    def supports_arg_streaming(self) -> bool:
        return False

    @override
    def decision_model(self, model: StepDecisionModel) -> StepDecisionModel | None:
        del model
        return None

    @override
    def prompt(self, spec: StepDecisionSpec, model: StepDecisionModel) -> str:
        return build_json_decision_prompt(spec, model) if spec.tools else ""

    @override
    def render_history(
        self, history: Sequence[HistoryItem], json_default: JsonDefault | None
    ) -> list[Message]:
        return render_prompt_json_history(history, json_default)

    @override
    def decode(
        self,
        response: LLMResponse,
        model: StepDecisionModel,
        tool_call_ids: ToolCallIdRegistry,
    ) -> ToolCallsDecision | None:
        output = _output(response, surrounding_text=True)
        if _decision_name(output) != "tool_calls":
            return None
        decision = model.validate(output, tool_call_ids)
        assert isinstance(decision, ToolCallsDecision)
        return decision


@final
class PromptJsonResultTransport(ResultTransport):
    @override
    def definitions(self, model: StepDecisionModel) -> list[ToolDefinition] | None:
        del model
        return None

    @override
    def decision_model(self, model: StepDecisionModel) -> StepDecisionModel | None:
        del model
        return None

    @override
    def prompt(self, spec: StepDecisionSpec, model: StepDecisionModel) -> str:
        return build_json_decision_prompt(spec, model) if not spec.tools else ""

    @override
    def decode(
        self, response: LLMResponse, model: StepDecisionModel
    ) -> ResultDecision | None:
        output = _output(response, surrounding_text=True)
        if model.mode is StepDecisionMode.RESULT_ONLY:
            assert model.result is not None
            return ResultDecision(model.result.validate(output))
        if _decision_name(output) != "result":
            return None
        decision = model.validate(output, None)
        assert isinstance(decision, ResultDecision)
        return decision

    @override
    def repair_messages(self, error: InvalidInferenceResponseError) -> list[Message]:
        return _repair_messages(error)


__all__ = [
    "EnvelopeToolCallTransport",
    "PromptJsonResultTransport",
    "PromptJsonToolCallTransport",
    "StructuredResultTransport",
]
