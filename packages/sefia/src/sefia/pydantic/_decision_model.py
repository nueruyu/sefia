from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional, Union, cast

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, create_model

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
from ._function_models import PydanticFunctionModelFactory


class _RuntimeToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class _ToolArgumentValidator:
    model_type: type[BaseModel]
    adapter: TypeAdapter


class PydanticDecisionModel(DecisionModel):
    def __init__(
        self,
        *,
        schema_model: type,
        runtime_model: type,
        mode: DecisionMode,
        argument_validators: dict[str, _ToolArgumentValidator],
    ):
        self._schema_adapter = TypeAdapter(schema_model)
        self._runtime_adapter = TypeAdapter(runtime_model)
        self._mode = mode
        self._argument_validators = argument_validators

    def schema(self) -> dict:
        schema = dict(self._schema_adapter.json_schema())
        schema["description"] = "The model for the LLM's decision on the next action."
        return schema

    def validate(self, data: Any) -> LLMDecision:
        try:
            decision = self._runtime_adapter.validate_python(data)
            tool_calls = self._validate_tool_calls(
                getattr(decision, "tool_calls", None)
            )
            final_answer = getattr(decision, "final_answer", None)
            return self._normalize_decision(
                final_answer=final_answer,
                tool_calls=tool_calls,
            )
        except ValidationError as e:
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

    def _validate_tool_calls(
        self, tool_calls: list[_RuntimeToolCall] | None
    ) -> list[DecisionToolCall] | None:
        if tool_calls is None:
            return None

        validated_calls: list[DecisionToolCall] = []
        for tool_call in tool_calls:
            validator = self._argument_validators.get(tool_call.name)
            if validator is None:
                arguments = dict(tool_call.arguments)
            else:
                try:
                    validated = validator.adapter.validate_python(tool_call.arguments)
                except ValidationError as e:
                    raise ValueError(
                        f"Tool arguments validation failed for {tool_call.name}: {e}"
                    ) from e
                arguments = cast(dict[str, Any], validated.model_dump())

            validated_calls.append(
                DecisionToolCall(name=tool_call.name, arguments=arguments)
            )
        return validated_calls


class _PydanticDecisionModelFactory:
    def __init__(
        self,
        function_model_factory: PydanticFunctionModelFactory | None = None,
    ):
        self._function_model_factory = (
            function_model_factory or PydanticFunctionModelFactory()
        )

    def build(self, spec: DecisionModelSpec) -> DecisionModel:
        argument_validators = self._build_argument_validators(spec.tools)
        schema_fields = self._schema_fields(spec, argument_validators)
        runtime_fields = self._runtime_fields(spec)
        return PydanticDecisionModel(
            schema_model=create_model(f"{spec.name}Schema", **schema_fields),
            runtime_model=create_model(f"{spec.name}Runtime", **runtime_fields),
            mode=spec.mode,
            argument_validators=argument_validators,
        )

    def _build_argument_validators(
        self, tools: list[Tool]
    ) -> dict[str, _ToolArgumentValidator]:
        validators = {}
        for tool in tools:
            model_type = self._function_model_factory.params_model(
                tool.function,
                name=tool.name,
                forbid_extra=True,
            )
            validators[tool.name] = _ToolArgumentValidator(
                model_type=model_type,
                adapter=TypeAdapter(model_type),
            )
        return validators

    def _schema_tool_calls_type(
        self,
        tools: list[Tool],
        argument_validators: dict[str, _ToolArgumentValidator],
    ) -> Any:
        call_models = [
            create_model(
                f"{tool.name}ToolCall",
                name=(Literal[tool.name], ...),
                arguments=(argument_validators[tool.name].model_type, ...),
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

    def _schema_fields(
        self,
        spec: DecisionModelSpec,
        argument_validators: dict[str, _ToolArgumentValidator],
    ) -> dict[str, Any]:
        if spec.mode is DecisionMode.TOOL_ONLY:
            return {
                "tool_calls": (
                    Optional[
                        self._schema_tool_calls_type(spec.tools, argument_validators)
                    ],
                    ...,
                )
            }
        if spec.mode is DecisionMode.TOOL_ENABLED:
            return {
                "final_answer": (Optional[spec.output_type], ...),
                "tool_calls": (
                    Optional[
                        self._schema_tool_calls_type(spec.tools, argument_validators)
                    ],
                    ...,
                ),
            }
        if spec.mode is DecisionMode.OUTPUT_ONLY:
            return {"final_answer": (spec.output_type, ...)}
        raise ValueError(f"Unsupported decision mode: {spec.mode!r}")

    def _runtime_fields(self, spec: DecisionModelSpec) -> dict[str, Any]:
        if spec.mode is DecisionMode.TOOL_ONLY:
            return {"tool_calls": (Optional[list[_RuntimeToolCall]], ...)}
        if spec.mode is DecisionMode.TOOL_ENABLED:
            return {
                "final_answer": (Optional[spec.output_type], ...),
                "tool_calls": (Optional[list[_RuntimeToolCall]], ...),
            }
        if spec.mode is DecisionMode.OUTPUT_ONLY:
            return {"final_answer": (spec.output_type, ...)}
        raise ValueError(f"Unsupported decision mode: {spec.mode!r}")
