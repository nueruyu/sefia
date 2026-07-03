from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, create_model

from .._interfaces.decision_model import (
    DecisionMode,
    DecisionModel,
    DecisionModelSpec,
    DecisionToolCall,
    LLMDecision,
    ResultLLMDecision,
    ToolCallsLLMDecision,
)
from .._tool_system import Tool
from ..exceptions import UnknownToolDecisionError
from ._function_models import PydanticFunctionModelFactory


class PydanticDecisionModel(DecisionModel):
    def __init__(
        self,
        *,
        model: Any,
    ):
        self._adapter = TypeAdapter(model)

    def schema(self) -> dict:
        schema = dict(self._adapter.json_schema())
        schema["description"] = "The model for the LLM's decision on the next action."
        return schema

    def validate(self, data: Any) -> LLMDecision:
        try:
            decision = self._adapter.validate_python(data)
            if decision.decision == "tool_calls":
                return ToolCallsLLMDecision(
                    tool_calls=self._extract_tool_calls(decision.tool_calls)
                )
            if decision.decision == "result":
                return ResultLLMDecision(result=decision.result)
            raise ValueError(f"Unsupported decision type: {decision.decision!r}")
        except ValidationError as e:
            unknown_tool_name = _unknown_tool_name_from_error(e)
            if unknown_tool_name is not None:
                raise UnknownToolDecisionError(unknown_tool_name) from e
            raise ValueError(f"Decision validation failed: {e}") from e

    def _extract_tool_calls(self, tool_calls: list[Any]) -> list[DecisionToolCall]:
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
            loc = item.get("loc")
            if not isinstance(loc, tuple) or "tool_calls" not in loc:
                continue
            ctx = item.get("ctx") or {}
            tag = ctx.get("tag")
            if isinstance(tag, str):
                return tag

        if item.get("type") == "literal_error":
            loc = item.get("loc")
            if isinstance(loc, tuple) and loc and loc[-1] == "name":
                value = item.get("input")
                if isinstance(value, str):
                    return value

    return None


class PydanticDecisionModelFactory:
    def __init__(
        self,
        function_model_factory: PydanticFunctionModelFactory | None = None,
    ):
        self._function_model_factory = (
            function_model_factory or PydanticFunctionModelFactory()
        )

    def build(self, spec: DecisionModelSpec) -> DecisionModel:
        return PydanticDecisionModel(
            model=self._model(spec),
        )

    def _tool_calls_type(self, tools: list[Tool]) -> Any:
        call_models = [
            create_model(
                f"{tool.name}ToolCall",
                name=(Literal[tool.name], ...),
                arguments=(
                    self._function_model_factory.params_model(
                        tool.schema,
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

    def _model(self, spec: DecisionModelSpec) -> Any:
        if spec.mode is DecisionMode.TOOL_ONLY:
            return self._tool_calls_model(spec)
        if spec.mode is DecisionMode.TOOL_ENABLED:
            branch_models = [
                self._tool_calls_model(spec),
                self._result_model(spec),
            ]
            return Annotated[
                Union[tuple(branch_models)],
                Field(discriminator="decision"),
            ]
        if spec.mode is DecisionMode.OUTPUT_ONLY:
            return self._result_model(spec)
        raise ValueError(f"Unsupported decision mode: {spec.mode!r}")

    def _tool_calls_model(self, spec: DecisionModelSpec) -> type:
        if spec.mode is DecisionMode.TOOL_ONLY:
            name = spec.name
        else:
            name = f"{spec.name}ToolCalls"

        return create_model(
            name,
            __config__=ConfigDict(extra="forbid"),
            decision=(Literal["tool_calls"], ...),
            tool_calls=(self._tool_calls_type(spec.tools), ...),
        )

    def _result_model(self, spec: DecisionModelSpec) -> type:
        if spec.mode is DecisionMode.OUTPUT_ONLY:
            name = spec.name
        else:
            name = f"{spec.name}Result"

        return create_model(
            name,
            __config__=ConfigDict(extra="forbid"),
            decision=(Literal["result"], ...),
            result=(spec.output_type, ...),
        )
