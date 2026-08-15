from __future__ import annotations

from typing import Annotated, Any, Literal, Union, cast

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, create_model
from typing_extensions import final, override

from .._interfaces.decision_model import (
    DecisionMode,
    DecisionModel,
    DecisionModelBuilder,
    DecisionModelSpec,
    DecisionToolCall,
    LLMDecision,
    ResultLLMDecision,
    ToolCallsLLMDecision,
)
from .._tool_system import JsonSchemaToolEntry, ToolEntry
from ..exceptions import UnknownToolDecisionError
from ..llm.schema import LLMSchema
from ._decision_schema import build_decision_schema
from ._tool_arguments import ToolArgumentContract, ToolSchemaKind


@final
class PydanticDecisionModel(DecisionModel):
    def __init__(
        self,
        *,
        model: Any,
        name: str,
        tools: dict[str, ToolArgumentContract],
    ):
        self._adapter = TypeAdapter(model)
        self._model = model
        self._name = name
        self._tools = tools
        self._schema: LLMSchema | None = None

    @override
    def schema(self) -> LLMSchema:
        if self._schema is None:
            self._schema = build_decision_schema(self._model, self._tools)
        return self._schema

    @override
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
            # ``arguments`` is the decoded, schema-validated dict (see
            # ``ToolArgumentContract``); no per-tool model to dump.
            calls.append(
                DecisionToolCall(
                    name=tool_call.name,
                    arguments=cast(dict[str, Any], tool_call.arguments),
                )
            )
        return calls


def _unknown_tool_name_from_error(error: ValidationError) -> str | None:
    for item in error.errors():
        if item.get("type") == "union_tag_invalid":
            loc = item.get("loc")
            if "tool_calls" not in loc:
                continue
            ctx = item.get("ctx") or {}
            tag = ctx.get("tag")
            if isinstance(tag, str):
                return tag

        if item.get("type") == "literal_error":
            loc = item.get("loc")
            if loc and loc[-1] == "name":
                value = item.get("input")
                if isinstance(value, str):
                    return value

    return None


@final
class PydanticDecisionModelFactory(DecisionModelBuilder):
    @override
    def build(self, spec: DecisionModelSpec) -> DecisionModel:
        tools = {
            tool.name: ToolArgumentContract(
                schema=tool.definition().parameters,
                kind=(
                    ToolSchemaKind.RAW
                    if isinstance(tool, JsonSchemaToolEntry)
                    else ToolSchemaKind.TYPED
                ),
            )
            for tool in spec.tools
        }
        return PydanticDecisionModel(
            model=self._model(spec, tools),
            name=spec.name,
            tools=tools,
        )

    def _tool_calls_type(
        self,
        tools: list[ToolEntry],
        contracts: dict[str, ToolArgumentContract],
    ) -> Any:
        call_models = [
            create_model(
                f"{tool.name}ToolCall",
                __config__=ConfigDict(extra="forbid"),
                name=(Literal[tool.name], ...),
                arguments=(
                    contracts[tool.name].validation_type(),
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

    def _model(
        self,
        spec: DecisionModelSpec,
        contracts: dict[str, ToolArgumentContract],
    ) -> Any:
        if spec.mode is DecisionMode.TOOL_ONLY:
            return self._tool_calls_model(spec, contracts)
        if spec.mode is DecisionMode.TOOL_ENABLED:
            branch_models = [
                self._tool_calls_model(spec, contracts),
                self._result_model(spec),
            ]
            return Annotated[
                Union[tuple(branch_models)],
                Field(discriminator="decision"),
            ]
        if spec.mode is DecisionMode.OUTPUT_ONLY:
            return self._result_model(spec)
        raise ValueError(f"Unsupported decision mode: {spec.mode!r}")

    def _tool_calls_model(
        self,
        spec: DecisionModelSpec,
        contracts: dict[str, ToolArgumentContract],
    ) -> type:
        if spec.mode is DecisionMode.TOOL_ONLY:
            name = spec.name
        else:
            name = f"{spec.name}ToolCalls"

        return create_model(
            name,
            __config__=ConfigDict(extra="forbid"),
            decision=(Literal["tool_calls"], ...),
            tool_calls=(self._tool_calls_type(spec.tools, contracts), ...),
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
