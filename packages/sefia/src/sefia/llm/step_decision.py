from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Never, Protocol, cast

import jsonschema.validators
from typing_extensions import final

from .._tool_system import JsonSchemaToolEntry, ToolEntry
from ..exceptions import UnknownToolDecisionError
from ..inference import ResultDecision, StepDecision, ToolCallRequest, ToolCallsDecision
from .json_schema import (
    DefinitionRegistry,
    JsonObject,
    JsonSchemaDocument,
    SchemaKeyword,
    SchemaPath,
)
from .structured_output import (
    StructuredOutputSchema,
    StructuredValue,
    StructuredValueSchema,
    StructuredValueSchemaFactory,
)

K = SchemaKeyword


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


class StepDecisionSchema(ABC):
    @property
    @abstractmethod
    def structured_output(self) -> StructuredOutputSchema: ...

    @abstractmethod
    def validate(
        self, value: StructuredValue, tool_call_ids: ToolCallIdSource | None
    ) -> StepDecision: ...


class StepDecisionSchemaFactory(ABC):
    @abstractmethod
    def create(self, spec: StepDecisionSpec) -> StepDecisionSchema: ...


class _ToolSchema:
    def __init__(self, document: JsonSchemaDocument, *, raw: bool):
        self.document = document
        self.raw = raw
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
class DefaultStepDecisionSchema(StepDecisionSchema):
    def __init__(
        self,
        spec: StepDecisionSpec,
        tools: dict[str, _ToolSchema],
        result: StructuredValueSchema | None,
    ):
        self._spec = spec
        self._tools = tools
        self._result = result
        self._structured_output = _build_structured_output(spec, tools, result)

    @property
    def structured_output(self) -> StructuredOutputSchema:
        return self._structured_output

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
class DefaultStepDecisionSchemaFactory(StepDecisionSchemaFactory):
    def __init__(self, value_schema_factory: StructuredValueSchemaFactory):
        self._value_schema_factory = value_schema_factory

    def create(self, spec: StepDecisionSpec) -> StepDecisionSchema:
        tools = {
            tool.name: _ToolSchema(
                JsonSchemaDocument.from_mapping(tool.definition().parameters),
                raw=isinstance(tool, JsonSchemaToolEntry),
            )
            for tool in spec.tools
        }
        result = (
            None
            if spec.mode is StepDecisionMode.TOOLS_REQUIRED
            else self._value_schema_factory.create(spec.output_type)
        )
        return DefaultStepDecisionSchema(spec, tools, result)


def _build_structured_output(
    spec: StepDecisionSpec,
    tools: dict[str, _ToolSchema],
    result: StructuredValueSchema | None,
) -> StructuredOutputSchema:
    definitions: JsonObject = {}
    registry = DefinitionRegistry(definitions)
    preserved: set[SchemaPath] = set()
    branches: list[tuple[JsonObject, set[SchemaPath]]] = []

    if spec.mode is not StepDecisionMode.RESULT_ONLY:
        branches.append(_tool_calls_branch(tools, registry))
    if spec.mode is not StepDecisionMode.TOOLS_REQUIRED:
        assert result is not None
        imported = registry.import_schema(
            result.json_schema.mutable_copy(), namespace="result"
        )
        branches.append(
            (
                _closed_object(
                    {
                        "decision": _literal("result"),
                        "result": imported.schema,
                    }
                ),
                set(),
            )
        )

    if len(branches) == 1:
        schema, preserved = branches[0]
    else:
        schema = cast(
            JsonObject,
            {
                K.ONE_OF: [branch for branch, _ in branches],
                "discriminator": {"propertyName": "decision"},
            },
        )
        for index, (_, paths) in enumerate(branches):
            for path in paths:
                if path and path[0] == K.DEFINITIONS:
                    preserved.add(path)
                else:
                    preserved.add((K.ONE_OF, index, *path))

    if definitions:
        schema[K.DEFINITIONS] = definitions
    schema[K.DESCRIPTION] = "The model for the LLM's decision on the next action."
    return StructuredOutputSchema(
        JsonSchemaDocument(schema),
        frozenset(preserved),
    )


def _tool_calls_branch(
    tools: dict[str, _ToolSchema], registry: DefinitionRegistry
) -> tuple[JsonObject, set[SchemaPath]]:
    call_schemas: list[JsonObject] = []
    raw_paths: set[SchemaPath] = set()
    raw_definitions: set[str] = set()
    multiple = len(tools) > 1

    for index, (name, tool) in enumerate(tools.items()):
        imported = registry.import_schema(tool.document.mutable_copy(), namespace=name)
        call_schemas.append(
            _closed_object(
                {
                    "name": _literal(name),
                    "arguments": imported.schema,
                }
            )
        )
        if tool.raw:
            registry.reserve(imported.definitions)
            prefix: SchemaPath = (K.ONE_OF, index) if multiple else ()
            raw_paths.add((*prefix, K.PROPERTIES, "arguments"))
            raw_definitions.update(imported.definitions)

    if multiple:
        items = cast(
            JsonObject,
            {
                K.ONE_OF: call_schemas,
                "discriminator": {"propertyName": "name"},
            },
        )
    else:
        items = call_schemas[0]
    branch = _closed_object(
        {
            "decision": _literal("tool_calls"),
            "tool_calls": {
                K.TYPE: "array",
                K.ITEMS: items,
                K.MIN_ITEMS: 1,
            },
        }
    )
    prefix = (K.PROPERTIES, "tool_calls", K.ITEMS)
    paths: set[SchemaPath] = {(*prefix, *path) for path in raw_paths}
    paths.update((K.DEFINITIONS, name) for name in raw_definitions)
    return branch, paths


def _closed_object(properties: JsonObject) -> JsonObject:
    return {
        K.TYPE: "object",
        K.PROPERTIES: properties,
        K.REQUIRED: list(properties),
        K.ADDITIONAL_PROPERTIES: False,
    }


def _literal(value: str) -> JsonObject:
    return {K.TYPE: "string", K.CONST: value}


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
    "DefaultStepDecisionSchemaFactory",
    "StepDecisionMode",
    "StepDecisionSchema",
    "StepDecisionSchemaFactory",
    "StepDecisionSpec",
    "ToolCallIdSource",
]
