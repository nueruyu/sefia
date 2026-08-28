from dataclasses import dataclass
from enum import Enum
from typing import Any, Never, cast

import jsonschema.validators
from typing_extensions import final

from .._tool_system import JsonSchemaToolEntry, ToolEntry
from ..exceptions import UnknownToolDecisionError
from ..inference import ResultDecision, StepDecision, ToolCallRequest, ToolCallsDecision
from .json_schema import JsonSchemaDocument
from ._tool_call_ids import ToolCallIdRegistry
from .result_format import ResultFormat, ResultFormatFactory
from .llm_output import LLMOutput


class StepDecisionMode(Enum):
    TOOLS_REQUIRED = "tools_required"
    TOOLS_OR_RESULT = "tools_or_result"
    RESULT_ONLY = "result_only"


class ToolSchemaSource(Enum):
    GENERATED = "generated"
    USER_DEFINED = "user_defined"


@dataclass(frozen=True)
class StepTool:
    name: str
    description: str
    arguments: JsonSchemaDocument
    schema_source: ToolSchemaSource


class _ToolModel:
    def __init__(self, step_tool: StepTool):
        self.schema = step_tool
        schema = step_tool.arguments.to_dict()
        validator_cls = jsonschema.validators.validator_for(
            schema, default=jsonschema.Draft202012Validator
        )
        validator_cls.check_schema(schema)
        self._validator = validator_cls(schema)

    def validate(self, value: LLMOutput) -> dict[str, Any]:
        value.to_object("arguments")
        value_dict = cast(dict[str, Any], value.data)
        errors = sorted(
            self._validator.iter_errors(value_dict),
            key=lambda error: list(error.path),
        )
        if errors:
            raise ValueError("; ".join(error.message for error in errors))
        return value_dict


@final
class DecisionSpec:
    @classmethod
    def for_inference(
        cls,
        *,
        output_type: Any,
        tools: list[ToolEntry],
        result_format_factory: ResultFormatFactory,
    ) -> "DecisionSpec":
        if output_type is Never:
            if not tools:
                raise ValueError(
                    "An @infer function returning Never must have tools available, "
                    "otherwise the inference loop can never make progress."
                )
            mode = StepDecisionMode.TOOLS_REQUIRED
        elif tools:
            mode = StepDecisionMode.TOOLS_OR_RESULT
        else:
            mode = StepDecisionMode.RESULT_ONLY
        return cls(
            output_type=output_type,
            tools=tools,
            mode=mode,
            result_format_factory=result_format_factory,
        )

    def __init__(
        self,
        *,
        output_type: Any,
        tools: list[ToolEntry],
        mode: StepDecisionMode,
        result_format_factory: ResultFormatFactory,
    ) -> None:
        if mode in (
            StepDecisionMode.TOOLS_REQUIRED,
            StepDecisionMode.TOOLS_OR_RESULT,
        ):
            if not tools:
                raise ValueError(f"{mode.value} decisions require at least one tool.")
        elif mode is StepDecisionMode.RESULT_ONLY:
            if tools:
                raise ValueError("result_only cannot include tools.")
        else:
            raise ValueError(f"Unsupported step decision mode: {mode!r}")

        prepared_tools: dict[str, _ToolModel] = {}
        for tool in tools:
            definition = tool.definition()
            prepared_tools[tool.name] = _ToolModel(
                StepTool(
                    name=tool.name,
                    description=definition.description,
                    arguments=JsonSchemaDocument.from_mapping(definition.parameters),
                    schema_source=(
                        ToolSchemaSource.USER_DEFINED
                        if isinstance(tool, JsonSchemaToolEntry)
                        else ToolSchemaSource.GENERATED
                    ),
                )
            )
        result = (
            None
            if mode is StepDecisionMode.TOOLS_REQUIRED
            else result_format_factory.create(output_type)
        )
        self._mode = mode
        self._tools = prepared_tools
        self._result = result

    @property
    def mode(self) -> StepDecisionMode:
        return self._mode

    @property
    def tools(self) -> tuple[StepTool, ...]:
        return tuple(tool.schema for tool in self._tools.values())

    @property
    def result(self) -> ResultFormat | None:
        return self._result

    def validate(
        self, value: LLMOutput, tool_call_ids: ToolCallIdRegistry | None
    ) -> StepDecision:
        try:
            data = value.to_object("step decision")
            decision = data.get("decision")
            decision_name = (
                decision.to_string("decision") if decision is not None else None
            )
            if decision_name == "tool_calls":
                if self._mode is StepDecisionMode.RESULT_ONLY:
                    raise ValueError("tool_calls is not allowed")
                return self._validate_tool_calls(data, tool_call_ids)
            if decision_name == "result":
                if self._mode is StepDecisionMode.TOOLS_REQUIRED:
                    raise ValueError("result is not allowed")
                return self._validate_result(data)
            raise ValueError("decision must be 'tool_calls' or 'result'")
        except UnknownToolDecisionError:
            raise
        except (TypeError, ValueError) as error:
            raise ValueError(f"Step decision validation failed: {error}") from error

    def _validate_tool_calls(
        self,
        data: dict[str, LLMOutput],
        tool_call_ids: ToolCallIdRegistry | None,
    ) -> ToolCallsDecision:
        _require_fields(data, {"decision", "tool_calls"})
        calls = data["tool_calls"].to_array("tool_calls")
        if not calls:
            raise ValueError("tool_calls must be a non-empty array")
        if tool_call_ids is None:
            raise RuntimeError("Tool call ids are required for tool calls.")

        requests: list[ToolCallRequest] = []
        for index, value in enumerate(calls):
            call = value.to_object("tool call")
            _require_fields(call, {"name", "arguments"})
            name = call["name"].to_string("tool name")
            tool = self._tools.get(name)
            if tool is None:
                raise UnknownToolDecisionError(name)
            requests.append(
                ToolCallRequest(
                    id=tool_call_ids.get_or_create(index),
                    name=name,
                    arguments=tool.validate(call["arguments"]),
                )
            )
        return ToolCallsDecision(requests)

    def _validate_result(self, data: dict[str, LLMOutput]) -> ResultDecision:
        _require_fields(data, {"decision", "result"})
        if self._result is None:
            raise ValueError("result is not allowed")
        return ResultDecision(self._result.validate(data["result"]))


def _require_fields(value: dict[str, LLMOutput], expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {sorted(missing)}")
        if extra:
            details.append(f"unexpected fields: {sorted(extra)}")
        raise ValueError(", ".join(details))


__all__ = [
    "DecisionSpec",
    "StepDecisionMode",
    "StepTool",
    "ToolSchemaSource",
]
