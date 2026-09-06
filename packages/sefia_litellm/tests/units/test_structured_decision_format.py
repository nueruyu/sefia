from dataclasses import dataclass
from typing import Annotated, Any, Literal, Never, cast

import pytest
from pydantic import Field
from sefia._tool_system import (
    JsonSchemaToolEntry,
    SignatureToolEntry,
    ToolEntry,
)
from sefia.llm.json_schema import SchemaNode
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.streaming import OutputStreamEvent, Scalar, StringDelta, StringEnd
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


def test_structured_decision_format_returns_defensive_schema_copies() -> None:
    decision_format = _prepare(_decision_model(str, [_tool()]))

    schema = decision_format.schema.to_dict()
    schema.clear()

    assert decision_format.schema.to_dict()


def test_tool_description_is_part_of_the_wire_schema() -> None:
    schema = _prepare(_decision_model(Never, [_tool()])).schema.to_dict()

    assert _tool_call_item(schema)["description"] == (
        "Ask the user a question and return the answer."
    )


def test_tool_without_description_omits_wire_schema_description() -> None:
    tool = _raw_tool(
        {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    )

    schema = _prepare(_decision_model(Never, [tool])).schema.to_dict()

    assert "description" not in _tool_call_item(schema)


def test_wire_schema_omits_openapi_discriminator_for_provider_compatibility() -> None:
    schema = _prepare(
        _decision_model(
            str,
            [
                _tool(),
                _raw_tool(
                    {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    }
                ),
            ],
        )
    ).schema.to_dict()

    assert all(
        "discriminator" not in cursor.node.value for cursor in SchemaNode(schema).walk()
    )


async def ask_user(question: Annotated[str, Field(min_length=1)]) -> str:
    """Ask the user a question and return the answer."""
    raise NotImplementedError


def _tool() -> ToolEntry:
    backend = PydanticModelBackend()
    name = backend.tool_name(ask_user)
    return SignatureToolEntry(
        ask_user,
        name=name,
        schema_source=ask_user,
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


def _name_constraint(name_schema: dict[str, Any]) -> Any:
    # A single Literal renders as `const`; multiple values render as `enum`.
    if "const" in name_schema:
        return name_schema["const"]
    return name_schema.get("enum")


def test_tool_only_schema_embeds_tool_argument_schema() -> None:
    definition = _decision_model(Never, [_tool()])

    schema = _prepare(definition).schema.to_dict()

    assert _tool_calls_array(schema)["minItems"] == 1
    item = _tool_call_item(schema)
    assert _name_constraint(item["properties"]["name"]) in ("ask_user", ["ask_user"])
    arguments = _resolve(item["properties"]["arguments"], schema)
    assert arguments["required"] == ["question"]
    assert arguments["additionalProperties"] is False
    assert arguments["properties"]["question"]["minLength"] == 1


def test_structured_decision_schema_always_uses_payload_envelope() -> None:
    specs = (
        _decision_model(Never, [_tool()]),
        _decision_model(str, [_tool()]),
        _decision_model(str, []),
    )

    schemas = [_prepare(spec).schema.to_dict() for spec in specs]

    for schema in schemas:
        assert schema["type"] == "object"
        assert schema["required"] == ["payload"]
        assert schema["additionalProperties"] is False
        assert "anyOf" not in schema
        assert set(cast(dict[str, Any], schema["properties"])) == {"payload"}
    assert "anyOf" not in _decision_schema(schemas[0])
    assert "anyOf" in _decision_schema(schemas[1])
    assert "anyOf" not in _decision_schema(schemas[2])


@pytest.mark.parametrize(
    "wire_data",
    [
        {"decision": "result", "result": "done"},
        {"payload": {"decision": "result", "result": "done"}, "extra": None},
    ],
)
def test_structured_decision_decode_requires_exact_payload_envelope(
    wire_data: Any,
) -> None:
    decision_format = _prepare(_decision_model(str, []))

    with pytest.raises(ValueError, match="structured decision envelope"):
        decision_format.decode(wire_data)


def test_result_shape_cannot_be_mistaken_for_a_tool_call() -> None:
    @dataclass
    class Output:
        name: Literal["ask_user"]
        arguments: dict[str, int]

    definition = _decision_model(Output, [_tool()])

    schema = _prepare(definition).schema.to_dict()
    result_branch = next(
        branch
        for branch in SchemaNode(_decision_schema(schema)).any_of()
        if branch.properties()["decision"].value["const"] == "result"
    )
    output = result_branch.properties()["result"]
    arguments = output.properties()["arguments"]
    assert arguments.type == "array"
    items = arguments.items()
    assert items is not None
    assert set(items.properties()) == {"key", "value"}


@pytest.mark.parametrize(
    ("wire", "logical"),
    [
        (StringEnd(("payload", "result"), "done"), StringEnd(("result",), "done")),
        (
            StringEnd(("payload", "tool_calls", 0, "name"), "lookup"),
            StringEnd(("tool_calls", 0, "name"), "lookup"),
        ),
        (
            StringDelta(("payload", "tool_calls", 0, "arguments", "key"), "it"),
            StringDelta(("tool_calls", 0, "arguments", "key"), "it"),
        ),
        (Scalar(("payload", "result", "count"), 1), Scalar(("result", "count"), 1)),
    ],
)
def test_stream_events_drop_payload_prefix(
    wire: OutputStreamEvent, logical: OutputStreamEvent
) -> None:
    decision_format = _prepare(_decision_model(str, [_tool()]))
    assert decision_format.decode_stream_event(wire) == logical
