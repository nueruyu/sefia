from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Never, Protocol, cast

import jsonschema.validators
from typing_extensions import final, override

from .._tool_system import JsonSchemaToolEntry, ToolEntry
from ..exceptions import UnknownToolDecisionError
from ..inference import ResultDecision, StepDecision, ToolCallRequest, ToolCallsDecision
from .json_schema import JsonSchemaDocument
from .structured_output import (
    StructuredValue,
    StructuredValueSchema,
    StructuredValueSchemaFactory,
)


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
    def tools_required(
        cls, *, name: str, output_type: Any, tools: list[ToolEntry]
    ) -> "StepDecisionSpec":
        return cls(name, output_type, tools, StepDecisionMode.TOOLS_REQUIRED)

    @classmethod
    def tools_or_result(
        cls, *, name: str, output_type: Any, tools: list[ToolEntry]
    ) -> "StepDecisionSpec":
        return cls(name, output_type, tools, StepDecisionMode.TOOLS_OR_RESULT)

    @classmethod
    def result_only(cls, *, name: str, output_type: Any) -> "StepDecisionSpec":
        return cls(name, output_type, [], StepDecisionMode.RESULT_ONLY)

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
            return cls.tools_required(name=name, output_type=output_type, tools=tools)
        if tools:
            return cls.tools_or_result(name=name, output_type=output_type, tools=tools)
        return cls.result_only(name=name, output_type=output_type)


class ToolCallIdSource(Protocol):
    def get_or_create(self, index: int) -> str: ...


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


class StepDecisionModel(ABC):
    @property
    @abstractmethod
    def mode(self) -> StepDecisionMode: ...

    @property
    @abstractmethod
    def tools(self) -> tuple[StepTool, ...]: ...

    @property
    @abstractmethod
    def result(self) -> StructuredValueSchema | None: ...

    @abstractmethod
    def validate(
        self, value: StructuredValue, tool_call_ids: ToolCallIdSource | None
    ) -> StepDecision: ...


class StepDecisionModelFactory(ABC):
    @abstractmethod
    def create(self, spec: StepDecisionSpec) -> StepDecisionModel: ...


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
        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            raise ValueError("arguments must be an object with string keys")
        value_dict = cast(dict[str, Any], value)
        errors = sorted(
            self._validator.iter_errors(value_dict),
            key=lambda error: list(error.path),
        )
        if errors:
            raise ValueError("; ".join(error.message for error in errors))
        return value_dict


@final
class DefaultStepDecisionModel(StepDecisionModel):
    def __init__(
        self,
        spec: StepDecisionSpec,
        tools: dict[str, _ToolModel],
        result: StructuredValueSchema | None,
    ):
        self._spec = spec
        self._tools = tools
        self._result = result

    @property
    @override
    def mode(self) -> StepDecisionMode:
        return self._spec.mode

    @property
    @override
    def tools(self) -> tuple[StepTool, ...]:
        return tuple(tool.schema for tool in self._tools.values())

    @property
    @override
    def result(self) -> StructuredValueSchema | None:
        return self._result

    def validate(
        self, value: StructuredValue, tool_call_ids: ToolCallIdSource | None
    ) -> StepDecision:
        try:
            data = _require_record(value, "step decision")
            decision = data.get("decision")
            if decision == "tool_calls":
                if self._spec.mode is StepDecisionMode.RESULT_ONLY:
                    raise ValueError("tool_calls is not allowed")
                return self._validate_tool_calls(data, tool_call_ids)
            if decision == "result":
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
        tool_call_ids: ToolCallIdSource | None,
    ) -> ToolCallsDecision:
        _require_fields(data, {"decision", "tool_calls"})
        calls = data["tool_calls"]
        if not isinstance(calls, list) or not calls:
            raise ValueError("tool_calls must be a non-empty array")
        if tool_call_ids is None:
            raise RuntimeError("Tool call ids are required for tool calls.")

        requests: list[ToolCallRequest] = []
        for index, value in enumerate(calls):
            call = _require_record(value, "tool call")
            _require_fields(call, {"name", "arguments"})
            name = call["name"]
            if not isinstance(name, str):
                raise ValueError("tool name must be a string")
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


@final
class DefaultStepDecisionModelFactory(StepDecisionModelFactory):
    def __init__(self, value_schema_factory: StructuredValueSchemaFactory):
        self._value_schema_factory = value_schema_factory

    def create(self, spec: StepDecisionSpec) -> StepDecisionModel:
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
            else self._value_schema_factory.create(spec.output_type)
        )
        return DefaultStepDecisionModel(spec, tools, result)


def _require_record(
    value: StructuredValue, description: str
) -> dict[str, StructuredValue]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{description} must be an object with string keys")
    return cast(dict[str, StructuredValue], value)


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
    "DefaultStepDecisionModelFactory",
    "JsonToolArguments",
    "StepDecisionMode",
    "StepDecisionModel",
    "StepDecisionModelFactory",
    "StepDecisionSpec",
    "StepTool",
    "ToolCallIdSource",
    "TypedToolArguments",
]
