from dataclasses import dataclass, make_dataclass
from typing import Any, Never, cast

import pytest
from sefia._tool_system import (
    JsonSchemaToolEntry,
    SignatureToolEntry,
    ToolEntry,
)
from sefia.inference import ResultDecision, StepDecision, ToolCallsDecision
from sefia.llm._tool_call_ids import ToolCallIdRegistry
from sefia.llm.json_schema import SchemaNode
from sefia.llm.step_decision import DecisionSpec
from sefia.pydantic import PydanticModelBackend
from sefia_litellm._schema import StructuredDecisionFormat


def _decision_model(output_type: Any, tools: list[ToolEntry]) -> DecisionSpec:
    return DecisionSpec.for_inference(
        output_type=output_type,
        tools=tools,
        result_format_factory=PydanticModelBackend(),
    )


def _prepare(decision: DecisionSpec):
    return StructuredDecisionFormat.from_spec(decision)


def _process(
    decision: DecisionSpec,
    data: Any,
    tool_call_ids: ToolCallIdRegistry | None = None,
) -> StepDecision:
    prepared = _prepare(decision)
    return decision.validate(prepared.decode({"payload": data}), tool_call_ids)


def _signature_tool(function: Any, *, name: str) -> ToolEntry:
    backend = PydanticModelBackend()
    return SignatureToolEntry(
        function,
        name=name,
        schema_source=function,
        inspector=backend,
    )


def _raw_tool(schema: dict[str, Any], *, name: str = "raw_tool") -> ToolEntry:
    async def handler(**kwargs: Any) -> str:
        return str(kwargs)

    return JsonSchemaToolEntry(handler, name=name, parameters=schema)


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follow a top-level ``$ref`` into ``$defs`` so assertions can inspect the
    embedded per-tool schemas regardless of how Pydantic hoists definitions."""
    if "$ref" in schema:
        key = schema["$ref"].split("/")[-1]
        return root["$defs"][key]
    return schema


def _decision_schema(schema: dict[str, Any]) -> dict[str, Any]:
    properties = cast(dict[str, Any], schema["properties"])
    payload = cast(dict[str, Any], properties["payload"])
    return _resolve(payload, schema)


def _tool_calls_array(schema: dict[str, Any]) -> dict[str, Any]:
    decision = _decision_schema(schema)
    tool_calls = decision["properties"]["tool_calls"]
    if "anyOf" in tool_calls:
        return next(
            candidate
            for candidate in tool_calls["anyOf"]
            if candidate.get("type") == "array"
        )
    return tool_calls


def _tool_call_item(schema: dict[str, Any]) -> dict[str, Any]:
    return _resolve(_tool_calls_array(schema)["items"], schema)


@dataclass
class _Audience:
    role: str


@dataclass
class _ArticleRequest:
    topic: str
    audience: _Audience


async def _research(article_request: _ArticleRequest) -> list[str]:
    return [article_request.topic]


def test_typed_tool_schema_hoists_nested_definitions() -> None:
    definition = _decision_model(Never, [_signature_tool(_research, name="research")])

    schema = _prepare(definition).schema.to_dict()

    arguments = _resolve(_tool_call_item(schema)["properties"]["arguments"], schema)
    request_schema = _resolve(arguments["properties"]["article_request"], schema)
    assert request_schema["required"] == ["topic", "audience"]
    assert request_schema["additionalProperties"] is False
    audience_schema = _resolve(request_schema["properties"]["audience"], schema)
    assert audience_schema["required"] == ["role"]


def test_raw_tool_schema_hoists_local_definitions() -> None:
    raw_schema = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}},
        "required": ["item"],
        "additionalProperties": False,
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            }
        },
    }
    definition = _decision_model(Never, [_raw_tool(raw_schema)])

    schema = _prepare(definition).schema.to_dict()
    arguments = _resolve(_tool_call_item(schema)["properties"]["arguments"], schema)

    assert "$defs" not in arguments
    assert arguments["properties"]["item"]["$ref"] == "#/$defs/tool_0__Item"
    item = SchemaNode(schema).definitions()["tool_0__Item"]
    assert item.properties()["name"].type == "string"
    assert item.strings("required") == ("name",)


def test_raw_definition_is_not_normalized_with_typed_decision_model() -> None:
    shared = make_dataclass("SharedPolicy", [("name", str)])

    async def typed(value: Any) -> None:
        pass

    typed.__annotations__["value"] = shared
    raw_schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/SharedPolicy"}},
        "required": ["value"],
        "additionalProperties": False,
        "$defs": {
            "SharedPolicy": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            }
        },
    }
    definition = _decision_model(
        Never,
        [_raw_tool(raw_schema), _signature_tool(typed, name="typed")],
    )

    with pytest.raises(ValueError, match=r"missing \['name'\]"):
        StructuredDecisionFormat.from_spec(definition)


def test_conflicting_tool_definition_names_are_renamed() -> None:
    first_type = make_dataclass("Shared", [("text", str)])
    second_type = make_dataclass("Shared", [("count", int)])

    async def first(value: Any) -> None:
        pass

    async def second(value: Any) -> None:
        pass

    first.__annotations__["value"] = first_type
    second.__annotations__["value"] = second_type
    definition = _decision_model(
        Never,
        [
            _signature_tool(first, name="first"),
            _signature_tool(second, name="second"),
        ],
    )

    schema = _prepare(definition).schema.to_dict()

    shared_definitions = {
        name: SchemaNode(schema).definitions()[name]
        for name in ("tool_0__Shared", "tool_1__Shared")
    }
    assert len(shared_definitions) == 2
    assert {
        tuple(definition.properties()) for definition in shared_definitions.values()
    } == {
        ("text",),
        ("count",),
    }
    root = SchemaNode(schema)
    for cursor in root.walk():
        if cursor.node.local_reference is not None:
            assert cursor.node.resolve_local_reference(root) is not None


def _result_schema(schema: dict[str, Any]) -> dict[str, Any]:
    decision = _decision_schema(schema)
    return _resolve(decision["properties"]["result"], schema)


@dataclass
class _Issue:
    description: str


@dataclass
class _Report:
    issues_by_perspective: dict[str, list[_Issue]]


def test_nested_mapping_result_is_lowered_and_decoded() -> None:
    definition = _decision_model(_Report, [])

    schema = _prepare(definition).schema.to_dict()
    report_schema = _result_schema(schema)
    report_schema = _resolve(report_schema, schema)
    mapping_schema = report_schema["properties"]["issues_by_perspective"]
    assert mapping_schema["type"] == "array"

    decision = _process(
        definition,
        {
            "decision": "result",
            "result": {
                "issues_by_perspective": [
                    {
                        "key": "Maintainability",
                        "value": [{"description": "Use clearer names."}],
                    }
                ]
            },
        },
    )

    assert isinstance(decision, ResultDecision)
    assert decision.result == _Report(
        issues_by_perspective={
            "Maintainability": [_Issue(description="Use clearer names.")]
        }
    )


async def _categorize(labels: dict[str, int]) -> None:
    pass


def test_mapping_tool_argument_is_lowered_and_decoded() -> None:
    definition = _decision_model(
        Never,
        [_signature_tool(_categorize, name="categorize")],
    )

    schema = _prepare(definition).schema.to_dict()
    arguments = _resolve(_tool_call_item(schema)["properties"]["arguments"], schema)
    assert arguments["properties"]["labels"]["type"] == "array"

    decision = _process(
        definition,
        {
            "decision": "tool_calls",
            "tool_calls": [
                {
                    "name": "categorize",
                    "arguments": {"labels": [{"key": "important", "value": 2}]},
                }
            ],
        },
        ToolCallIdRegistry(),
    )

    assert isinstance(decision, ToolCallsDecision)
    assert decision.calls[0].arguments == {"labels": {"important": 2}}
