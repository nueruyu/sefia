from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    WithJsonSchema,
    create_model,
)
from typing_extensions import final, override

from ..llm.step_decision import (
    StepDecisionMode,
    StepDecisionSchema,
    StepDecisionSchemaFactory,
    StepDecisionSpec,
    ToolCallIdSource,
)
from .._tool_system import JsonSchemaToolEntry, ToolEntry
from ..exceptions import UnknownToolDecisionError
from ..inference import ResultDecision, StepDecision, ToolCallRequest, ToolCallsDecision
from ..llm.json_schema import (
    DefinitionRegistry,
    JsonSchemaDocument,
    SchemaKeyword,
    SchemaNode,
    SchemaPath,
    require_json_object,
)
from ..llm.structured_output import StructuredOutputSchema, StructuredValue
from ._tool_arguments import (
    ToolArgumentContract,
    ToolSchemaKind,
)

_TOOL_ARGUMENT_MARKER = "x-sefia-tool-arguments"
K = SchemaKeyword


class _ToolCallPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any]


class _ToolCallsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["tool_calls"]
    tool_calls: list[_ToolCallPayload]


class _ResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["result"]
    result: object


_StepDecisionPayload = _ToolCallsPayload | _ResultPayload


def _tool_argument_schema_placeholder(tool_name: str) -> WithJsonSchema:
    return WithJsonSchema({_TOOL_ARGUMENT_MARKER: tool_name})


def _compose_structured_output_schema(
    model: Any, tools: dict[str, ToolArgumentContract]
) -> StructuredOutputSchema:
    schema = require_json_object(TypeAdapter(model).json_schema())
    root = SchemaNode(schema)
    registry = DefinitionRegistry(root.ensure_definitions())
    raw_paths: set[SchemaPath] = set()

    for cursor in list(root.walk()):
        tool_name = cursor.node.value.pop(_TOOL_ARGUMENT_MARKER, None)
        if not isinstance(tool_name, str):
            continue
        if tool_name not in tools:
            raise ValueError(f"Unknown tool argument marker: {tool_name!r}")

        contract = tools[tool_name]
        imported = registry.import_schema(
            contract.schema.mutable_copy(),
            namespace=tool_name,
        )
        cursor.node.value.clear()
        cursor.node.value.update(imported.schema)
        if contract.kind is ToolSchemaKind.RAW:
            registry.reserve(imported.definitions)
            raw_paths.add(cursor.path)
            raw_paths.update((K.DEFINITIONS, name) for name in imported.definitions)

    root.set_description("The model for the LLM's decision on the next action.")
    return StructuredOutputSchema(
        document=JsonSchemaDocument.from_mapping(schema),
        preserved_schema_paths=frozenset(raw_paths),
    )


@final
class PydanticStepDecisionSchema(StepDecisionSchema):
    def __init__(self, model: Any, structured_output: StructuredOutputSchema):
        self._adapter: TypeAdapter[_StepDecisionPayload] = TypeAdapter(model)
        self._structured_output = structured_output

    @property
    @override
    def structured_output(self) -> StructuredOutputSchema:
        return self._structured_output

    @override
    def validate(
        self, value: StructuredValue, tool_call_ids: ToolCallIdSource | None
    ) -> StepDecision:
        try:
            payload = self._adapter.validate_python(value)
            if isinstance(payload, _ToolCallsPayload):
                if tool_call_ids is None:
                    raise RuntimeError("Tool call ids are required for tool calls.")
                return ToolCallsDecision(
                    calls=[
                        ToolCallRequest(
                            id=tool_call_ids.get_or_create(index),
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                        )
                        for index, tool_call in enumerate(payload.tool_calls)
                    ]
                )
            return ResultDecision(result=payload.result)
        except ValidationError as e:
            unknown_tool_name = _unknown_tool_name_from_error(e)
            if unknown_tool_name is not None:
                raise UnknownToolDecisionError(unknown_tool_name) from e
            raise ValueError(f"Step decision validation failed: {e}") from e


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
class PydanticStepDecisionSchemaFactory(StepDecisionSchemaFactory):
    @override
    def create(self, spec: StepDecisionSpec) -> StepDecisionSchema:
        tools = {
            tool.name: ToolArgumentContract(
                schema=JsonSchemaDocument.from_mapping(tool.definition().parameters),
                kind=(
                    ToolSchemaKind.RAW
                    if isinstance(tool, JsonSchemaToolEntry)
                    else ToolSchemaKind.TYPED
                ),
            )
            for tool in spec.tools
        }
        model = self._model(spec, tools)
        return PydanticStepDecisionSchema(
            model,
            _compose_structured_output_schema(model, tools),
        )

    def _tool_calls_type(
        self,
        tools: list[ToolEntry],
        contracts: dict[str, ToolArgumentContract],
    ) -> Any:
        call_models = [
            create_model(
                f"{tool.name}ToolCall",
                __base__=_ToolCallPayload,
                name=(Literal[tool.name], ...),
                arguments=(
                    Annotated[
                        contracts[tool.name].validation_type(),
                        _tool_argument_schema_placeholder(tool.name),
                    ],
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
        spec: StepDecisionSpec,
        contracts: dict[str, ToolArgumentContract],
    ) -> Any:
        if spec.mode is StepDecisionMode.TOOLS_REQUIRED:
            return self._tool_calls_model(spec, contracts)
        if spec.mode is StepDecisionMode.TOOLS_OR_RESULT:
            branch_models = [
                self._tool_calls_model(spec, contracts),
                self._result_model(spec),
            ]
            return Annotated[
                Union[tuple(branch_models)],
                Field(discriminator="decision"),
            ]
        if spec.mode is StepDecisionMode.RESULT_ONLY:
            return self._result_model(spec)
        raise ValueError(f"Unsupported step decision mode: {spec.mode!r}")

    def _tool_calls_model(
        self,
        spec: StepDecisionSpec,
        contracts: dict[str, ToolArgumentContract],
    ) -> type[_ToolCallsPayload]:
        if spec.mode is StepDecisionMode.TOOLS_REQUIRED:
            name = spec.name
        else:
            name = f"{spec.name}ToolCalls"

        return create_model(
            name,
            __base__=_ToolCallsPayload,
            tool_calls=(self._tool_calls_type(spec.tools, contracts), ...),
        )

    def _result_model(self, spec: StepDecisionSpec) -> type[_ResultPayload]:
        if spec.mode is StepDecisionMode.RESULT_ONLY:
            name = spec.name
        else:
            name = f"{spec.name}Result"

        return create_model(
            name,
            __base__=_ResultPayload,
            result=(spec.output_type, ...),
        )
