from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import Field, TypeAdapter, ValidationError, create_model

from .._interfaces.decision_model import (
    DecisionMode,
    DecisionModel,
    DecisionModelSpec,
    DecisionToolCall,
    FinalAnswerLLMDecision,
    LLMDecision,
    ToolCallsLLMDecision,
)
from .._tool_system import Tool
from ..exceptions import UnknownToolDecisionError
from ._function_models import PydanticFunctionModelFactory


class PydanticDecisionModel(DecisionModel):
    def __init__(
        self,
        *,
        model: type,
        mode: DecisionMode,
    ):
        self._adapter = TypeAdapter(model)
        self._mode = mode

    def schema(self) -> dict:
        schema = dict(self._adapter.json_schema())
        schema["description"] = "The model for the LLM's decision on the next action."
        return schema

    def validate(self, data: Any) -> LLMDecision:
        try:
            decision = self._adapter.validate_python(data)
            tool_calls = self._extract_tool_calls(getattr(decision, "tool_calls", None))
            final_answer = getattr(decision, "final_answer", None)
            return self._normalize_decision(
                final_answer=final_answer,
                tool_calls=tool_calls,
            )
        except ValidationError as e:
            unknown_tool_name = _unknown_tool_name_from_error(e)
            if unknown_tool_name is not None:
                raise UnknownToolDecisionError(unknown_tool_name) from e
            raise ValueError(f"Decision validation failed: {e}") from e

    def _normalize_decision(
        self,
        *,
        final_answer: Any,
        tool_calls: list[DecisionToolCall] | None,
    ) -> LLMDecision:
        has_tool_calls = bool(tool_calls)
        has_final_answer = final_answer is not None

        if self._mode is DecisionMode.TOOL_ONLY:
            if not has_tool_calls:
                raise ValueError("Decision must contain 'tool_calls'.")
            return ToolCallsLLMDecision(tool_calls=tool_calls)

        if self._mode is DecisionMode.OUTPUT_ONLY:
            return FinalAnswerLLMDecision(final_answer=final_answer)

        if self._mode is DecisionMode.TOOL_ENABLED:
            if has_tool_calls and has_final_answer:
                raise ValueError(
                    "Decision must not contain both 'tool_calls' and 'final_answer'."
                )
            if has_tool_calls:
                return ToolCallsLLMDecision(tool_calls=tool_calls)
            if has_final_answer:
                return FinalAnswerLLMDecision(final_answer=final_answer)
            raise ValueError(
                "Decision must contain either 'tool_calls' or 'final_answer'."
            )

        raise ValueError(f"Unsupported decision mode: {self._mode!r}")

    def _extract_tool_calls(
        self, tool_calls: list[Any] | None
    ) -> list[DecisionToolCall] | None:
        if tool_calls is None:
            return None

        calls: list[DecisionToolCall] = []
        for tool_call in tool_calls:
            calls.append(
                DecisionToolCall(
                    name=tool_call.name,
                    arguments=tool_call.arguments.model_dump(),
                )
            )
        return calls


def _unknown_tool_name_from_error(error: ValidationError) -> str | None:
    for item in error.errors():
        if item.get("type") == "union_tag_invalid":
            ctx = item.get("ctx") or {}
            tag = ctx.get("tag")
            if isinstance(tag, str):
                return tag

        if item.get("type") == "literal_error":
            loc = item.get("loc")
            if isinstance(loc, tuple) and loc[-1] == "name":
                value = item.get("input")
                if isinstance(value, str):
                    return value

    return None


class _PydanticDecisionModelFactory:
    def __init__(
        self,
        function_model_factory: PydanticFunctionModelFactory | None = None,
    ):
        self._function_model_factory = (
            function_model_factory or PydanticFunctionModelFactory()
        )

    def build(self, spec: DecisionModelSpec) -> DecisionModel:
        fields = self._fields(spec)
        return PydanticDecisionModel(
            model=create_model(spec.name, **fields),
            mode=spec.mode,
        )

    def _tool_calls_type(self, tools: list[Tool]) -> Any:
        call_models = [
            create_model(
                f"{tool.name}ToolCall",
                name=(Literal[tool.name], ...),
                arguments=(
                    self._function_model_factory.params_model(
                        tool.function,
                        name=tool.name,
                        forbid_extra=True,
                    ),
                    ...,
                ),
            )
            for tool in tools
        ]
        if len(call_models) == 1:
            item_type: Any = call_models[0]
        else:
            item_type = Annotated[
                Union[tuple(call_models)], Field(discriminator="name")
            ]
        return Annotated[list[item_type], Field(min_length=1)]

    def _fields(self, spec: DecisionModelSpec) -> dict[str, Any]:
        if spec.mode is DecisionMode.TOOL_ONLY:
            return {
                "tool_calls": (
                    Optional[self._tool_calls_type(spec.tools)],
                    ...,
                )
            }
        if spec.mode is DecisionMode.TOOL_ENABLED:
            return {
                "final_answer": (Optional[spec.output_type], ...),
                "tool_calls": (
                    Optional[self._tool_calls_type(spec.tools)],
                    ...,
                ),
            }
        if spec.mode is DecisionMode.OUTPUT_ONLY:
            return {"final_answer": (spec.output_type, ...)}
        raise ValueError(f"Unsupported decision mode: {spec.mode!r}")
