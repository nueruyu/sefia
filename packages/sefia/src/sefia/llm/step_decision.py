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
from .result_schema import ResultSchema, ResultSchemaFactory
from .structured_value import StructuredValue


class StepDecisionMode(Enum):
    TOOLS_REQUIRED = "tools_required"
    TOOLS_OR_RESULT = "tools_or_result"
    RESULT_ONLY = "result_only"


@dataclass(frozen=True)
class StepDecisionSpec:
    name: str
    output_type: Any
    tools: list[ToolEntry]
    mode: StepDecisionMode

    def __post_init__(self) -> None:
        if self.mode in (
            StepDecisionMode.TOOLS_REQUIRED,
            StepDecisionMode.TOOLS_OR_RESULT,
        ):
            if not self.tools:
                raise ValueError(
                    f"{self.mode.value} decisions require at least one tool."
                )
        elif self.mode is StepDecisionMode.RESULT_ONLY:
            if self.tools:
                raise ValueError("result_only cannot include tools.")
        else:
            raise ValueError(f"Unsupported step decision mode: {self.mode!r}")

    @classmethod
    def for_inference(
        cls, *, name: str, output_type: Any, tools: list[ToolEntry]
    ) -> "StepDecisionSpec":
        if output_type is Never:
            if not tools:
                raise ValueError(
                    "An @infer function returning Never must have tools available, "
                    "otherwise the inference loop can never make progress."
                )
            return cls(name, output_type, tools, StepDecisionMode.TOOLS_REQUIRED)
        if tools:
            return cls(name, output_type, tools, StepDecisionMode.TOOLS_OR_RESULT)
        return cls(name, output_type, [], StepDecisionMode.RESULT_ONLY)


@dataclass(frozen=True)
class TypedToolArguments:
    json_schema: JsonSchemaDocument


@dataclass(frozen=True)
class JsonToolArguments:
    json_schema: JsonSchemaDocument


ToolArguments = TypedToolArguments | JsonToolArguments


@dataclass(frozen=True)
class StepTool:
    name: str
    arguments: ToolArguments


class _ToolModel:
    def __init__(self, step_tool: StepTool):
        self.schema = step_tool
        document = step_tool.arguments.json_schema
        schema = document.to_dict()
        validator_cls = jsonschema.validators.validator_for(
            schema, default=jsonschema.Draft202012Validator
        )
        validator_cls.check_schema(schema)
        self._validator = validator_cls(schema)

    def validate(self, value: StructuredValue) -> dict[str, Any]:
        value.to_object("arguments")
        value_dict = cast(dict[str, Any], value.value)
        errors = sorted(
            self._validator.iter_errors(value_dict),
            key=lambda error: list(error.path),
        )
        if errors:
            raise ValueError("; ".join(error.message for error in errors))
        return value_dict


@final
class StepDecisionModel:
    @classmethod
    def from_spec(
        cls,
        spec: StepDecisionSpec,
        result_schema_factory: ResultSchemaFactory,
    ) -> "StepDecisionModel":
        tools = {
            tool.name: _ToolModel(
                StepTool(
                    name=tool.name,
                    arguments=(
                        JsonToolArguments(
                            JsonSchemaDocument.from_mapping(
                                tool.definition().parameters
                            )
                        )
                        if isinstance(tool, JsonSchemaToolEntry)
                        else TypedToolArguments(
                            JsonSchemaDocument.from_mapping(
                                tool.definition().parameters
                            )
                        )
                    ),
                )
            )
            for tool in spec.tools
        }
        result = (
            None
            if spec.mode is StepDecisionMode.TOOLS_REQUIRED
            else result_schema_factory.create(spec.output_type)
        )
        return cls(spec, tools, result)

    def __init__(
        self,
        spec: StepDecisionSpec,
        tools: dict[str, _ToolModel],
        result: ResultSchema | None,
    ):
        self._spec = spec
        self._tools = tools
        self._result = result

    @property
    def mode(self) -> StepDecisionMode:
        return self._spec.mode

    @property
    def tools(self) -> tuple[StepTool, ...]:
        return tuple(tool.schema for tool in self._tools.values())

    @property
    def result(self) -> ResultSchema | None:
        return self._result

    def validate(
        self, value: StructuredValue, tool_call_ids: ToolCallIdRegistry | None
    ) -> StepDecision:
        try:
            data = value.to_object("step decision")
            decision = data.get("decision")
            decision_name = (
                decision.to_string("decision") if decision is not None else None
            )
            if decision_name == "tool_calls":
                if self._spec.mode is StepDecisionMode.RESULT_ONLY:
                    raise ValueError("tool_calls is not allowed")
                return self._validate_tool_calls(data, tool_call_ids)
            if decision_name == "result":
                if self._spec.mode is StepDecisionMode.TOOLS_REQUIRED:
                    raise ValueError("result is not allowed")
                return self._validate_result(data)
            raise ValueError("decision must be 'tool_calls' or 'result'")
        except UnknownToolDecisionError:
            raise
        except (TypeError, ValueError) as error:
            raise ValueError(f"Step decision validation failed: {error}") from error

    def _validate_tool_calls(
        self,
        data: dict[str, StructuredValue],
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

    def _validate_result(self, data: dict[str, StructuredValue]) -> ResultDecision:
        _require_fields(data, {"decision", "result"})
        if self._result is None:
            raise ValueError("result is not allowed")
        return ResultDecision(self._result.validate(data["result"]))


def _require_fields(value: dict[str, StructuredValue], expected: set[str]) -> None:
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
    "JsonToolArguments",
    "StepDecisionMode",
    "StepDecisionModel",
    "StepDecisionSpec",
    "StepTool",
    "TypedToolArguments",
]
