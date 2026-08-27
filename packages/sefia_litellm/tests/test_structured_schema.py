import json
from dataclasses import dataclass, make_dataclass
from typing import Annotated, Any, Literal, Never, cast
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, ConfigDict, Field

from sefia._tool_system import (
    JsonSchemaToolEntry,
    SignatureToolEntry,
    ToolEntry,
    ToolRegistry,
)
from sefia.event_system import Event, EventPublisher
from sefia.exceptions import InvalidInferenceResponseError, UnknownToolDecisionError
from sefia.inference import FunctionInfo, StepDecision, ToolCallsDecision
from sefia.inference import ResultDecision
from sefia.llm import LLMClient, LLMInferenceStrategy, LLMResponse
from sefia.llm.json_schema import SchemaNode
from sefia.llm.streaming import OutputStreamEvent
from sefia.llm.llm_output import LLMOutput
from sefia.llm.step_decision import (
    StepDecisionMode,
    StepDecisionModel,
    StepDecisionSpec,
)
from sefia.llm._tool_call_ids import ToolCallIdRegistry
from sefia.llm._arg_stream import ToolArgStreamer
from sefia.pydantic import PydanticModelBackend
from sefia.streaming import ArgStream, StringEnd
from sefia_litellm._schema import DecisionEnvelope, DecisionEnvelopeFormat
from sefia_litellm._output_stream import OutputEventStreamer


def _decision_model(output_type: Any, tools: list[ToolEntry]) -> StepDecisionModel:
    spec = StepDecisionSpec.for_inference(
        name="StepDecision", output_type=output_type, tools=tools
    )
    return StepDecisionModel.from_spec(spec, PydanticModelBackend())


def _prepare(decision: StepDecisionModel):
    return DecisionEnvelopeFormat.from_model(decision)


def _process(
    decision: StepDecisionModel,
    data: Any,
    tool_call_ids: ToolCallIdRegistry | None = None,
) -> StepDecision:
    prepared = _prepare(decision)
    return decision.validate(prepared.decode(data), tool_call_ids)


def test_envelope_format_removes_payload_from_stream_paths() -> None:
    envelope_format = _prepare(_decision_model(str, []))

    assert envelope_format.to_payload_path(
        ("payload", "tool_calls", 0, "arguments", "question")
    ) == ("tool_calls", 0, "arguments", "question")


def test_decision_envelope_models_the_wire_payload() -> None:
    payload = LLMOutput.from_json({"decision": "result", "result": "done"})

    envelope = DecisionEnvelope.from_output(LLMOutput.from_object({"payload": payload}))

    assert envelope.payload.data == payload.data


def test_decision_envelope_rejects_other_fields() -> None:
    output = LLMOutput.from_json({"payload": {}, "extra": True})

    with pytest.raises(ValueError, match="exactly one payload"):
        DecisionEnvelope.from_output(output)


def test_decision_envelope_format_returns_defensive_schema_copies() -> None:
    envelope_format = _prepare(_decision_model(str, [_tool()]))

    schema = envelope_format.schema.to_dict()
    schema.clear()

    assert envelope_format.schema.to_dict()


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


def test_generated_schema_titles_are_removed() -> None:
    schema = _prepare(_decision_model(str, [_tool()])).schema.to_dict()

    assert all("title" not in cursor.node.value for cursor in SchemaNode(schema).walk())
    assert "description" not in schema


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


async def test_payload_stream_reaches_preview_as_a_logical_argument() -> None:
    prepared = _prepare(_decision_model(Never, [_tool()]))
    events: list[object] = []

    async def collect(_tool_call_id: str, stream: ArgStream) -> None:
        async for event in stream:
            events.append(event)

    streamer = ToolArgStreamer(
        {"ask_user": collect},
        lambda index: f"call-{index}",
    )
    wire_streamer = OutputEventStreamer(
        prepared, lambda event: _dispatch_event(streamer, event)
    )
    await wire_streamer.feed(
        '{"payload":{"decision":"tool_calls","tool_calls":['
        '{"name":"ask_user","arguments":{"question":"Hello"}}]}}'
    )
    await streamer.close()

    assert StringEnd(name="question", value="Hello") in events


async def _dispatch_event(streamer: ToolArgStreamer, event: OutputStreamEvent) -> None:
    streamer.on_event(event)


async def ask_user(question: Annotated[str, Field(min_length=1)]) -> str:
    """Ask the user a question and return the answer."""
    raise NotImplementedError


class _MockPublisher(EventPublisher):
    def __init__(self) -> None:
        super().__init__(handlers=[])

    async def publish(self, event: Event) -> None:
        pass


def _tool() -> ToolEntry:
    backend = PydanticModelBackend()
    name = backend.tool_name(ask_user)
    return SignatureToolEntry(
        ask_user,
        name=name,
        schema_source=ask_user,
        inspector=backend,
    )


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


def _tool_calls_array(schema: dict[str, Any]) -> dict[str, Any]:
    payload = _resolve(cast(dict[str, Any], schema["properties"]["payload"]), schema)
    tool_calls = payload["properties"]["tool_calls"]
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


def test_result_shape_cannot_be_mistaken_for_a_tool_call() -> None:
    @dataclass
    class Output:
        name: Literal["ask_user"]
        arguments: dict[str, int]

    definition = _decision_model(Output, [_tool()])

    schema = _prepare(definition).schema.to_dict()
    payload = SchemaNode(schema).properties()["payload"].value

    result_branch = next(
        branch
        for branch in SchemaNode(payload).any_of()
        if branch.properties()["decision"].value["const"] == "result"
    )
    output = result_branch.properties()["result"]
    arguments = output.properties()["arguments"]
    assert arguments.type == "array"
    items = arguments.items()
    assert items is not None
    assert set(items.properties()) == {"key", "value"}


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
        DecisionEnvelopeFormat.from_model(definition)


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
    _assert_local_references_resolve(schema)


def _assert_local_references_resolve(
    node: Any, root: dict[str, Any] | None = None
) -> None:
    if root is None:
        assert isinstance(node, dict)
        root = cast(dict[str, Any], node)
    if isinstance(node, list):
        for item in cast(list[Any], node):
            _assert_local_references_resolve(item, root)
        return
    if not isinstance(node, dict):
        return
    node_dict = cast(dict[str, Any], node)
    reference = node_dict.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        assert (
            SchemaNode(node_dict).resolve_local_reference(SchemaNode(root)) is not None
        )
    for value in node_dict.values():
        _assert_local_references_resolve(value, root)


def test_transitive_definition_collisions_remain_fragment_local() -> None:
    def fragment(value_type: str) -> dict[str, Any]:
        return {
            "$ref": "#/$defs/A",
            "$defs": {
                "A": {
                    "type": "object",
                    "properties": {"value": {"$ref": "#/$defs/B"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                "B": {"$ref": "#/$defs/C"},
                "C": {"type": value_type},
            },
        }

    definition = _decision_model(
        Never,
        [
            _raw_tool(fragment("string"), name="first"),
            _raw_tool(fragment("integer"), name="second"),
        ],
    )

    schema = _prepare(definition).schema.to_dict()
    definitions = SchemaNode(schema).definitions()

    assert definitions["tool_0__A"].properties()["value"].reference == (
        "#/$defs/tool_0__B"
    )
    assert definitions["tool_0__B"].reference == "#/$defs/tool_0__C"
    assert definitions["tool_0__C"].type == "string"
    assert definitions["tool_1__A"].properties()["value"].reference == (
        "#/$defs/tool_1__B"
    )
    assert definitions["tool_1__B"].reference == "#/$defs/tool_1__C"
    assert definitions["tool_1__C"].type == "integer"
    _assert_local_references_resolve(schema)

    first = _process(
        definition,
        {
            "payload": {
                "decision": "tool_calls",
                "tool_calls": [{"name": "first", "arguments": {"value": "one"}}],
            }
        },
        ToolCallIdRegistry(),
    )
    second = _process(
        definition,
        {
            "payload": {
                "decision": "tool_calls",
                "tool_calls": [{"name": "second", "arguments": {"value": 2}}],
            }
        },
        ToolCallIdRegistry(),
    )
    assert isinstance(first, ToolCallsDecision)
    assert first.calls[0].arguments == {"value": "one"}
    assert isinstance(second, ToolCallsDecision)
    assert second.calls[0].arguments == {"value": 2}


def test_identical_definitions_are_namespaced_deterministically() -> None:
    fragment = {
        "$ref": "#/$defs/Item",
        "$defs": {"Item": {"type": "string"}},
    }
    definition = _decision_model(
        Never,
        [
            _raw_tool(fragment, name="first"),
            _raw_tool(fragment, name="second"),
        ],
    )

    first_schema = _prepare(definition).schema.to_dict()
    second_schema = _prepare(definition).schema.to_dict()

    assert first_schema == second_schema
    definitions = SchemaNode(first_schema).definitions()
    assert definitions["tool_0__Item"].type == "string"
    assert definitions["tool_1__Item"].type == "string"
    assert fragment == {
        "$ref": "#/$defs/Item",
        "$defs": {"Item": {"type": "string"}},
    }


def test_compatible_raw_tool_schema_is_preserved_verbatim() -> None:
    raw_schema = {
        "title": "SearchArguments",
        "type": "object",
        "properties": {"query": {"title": "Query", "type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    definition = _decision_model(Never, [_raw_tool(raw_schema)])

    schema = _prepare(definition).schema.to_dict()

    arguments = _resolve(_tool_call_item(schema)["properties"]["arguments"], schema)
    assert arguments == raw_schema


@pytest.mark.parametrize(
    ("raw_schema", "message"),
    [
        (
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "additionalProperties": False,
            },
            "all object properties must be required",
        ),
        (
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            "additionalProperties to false",
        ),
        (
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": True,
            },
            "additionalProperties to false",
        ),
        (
            {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "additionalProperties to false",
        ),
        (
            {
                "oneOf": [
                    {"type": "object", "additionalProperties": False},
                    {"type": "object", "additionalProperties": False},
                ]
            },
            "oneOf is not supported",
        ),
    ],
)
def test_incompatible_raw_tool_schema_is_rejected_without_rewriting(
    raw_schema: dict[str, Any], message: str
) -> None:
    definition = _decision_model(Never, [_raw_tool(raw_schema)])

    with pytest.raises(ValueError, match=message):
        _prepare(definition)


_UNSUPPORTED_COMPOSITIONS: list[tuple[str, Any]] = [
    ("allOf", [{}]),
    ("not", {}),
    ("dependentRequired", {"query": ["other"]}),
    ("dependentSchemas", {"query": {}}),
    ("if", {}),
    ("then", {}),
    ("else", {}),
]


@pytest.mark.parametrize(
    ("keyword", "value"),
    _UNSUPPORTED_COMPOSITIONS,
)
def test_unsupported_composition_keyword_is_rejected(keyword: str, value: Any) -> None:
    raw_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
        keyword: value,
    }
    definition = _decision_model(Never, [_raw_tool(raw_schema)])

    with pytest.raises(ValueError, match=rf"{keyword} is not supported"):
        _prepare(definition)


@pytest.mark.parametrize(
    "property_name",
    [
        "allOf",
        "not",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
        "oneOf",
    ],
)
def test_schema_keyword_is_allowed_as_property_name(property_name: str) -> None:
    raw_schema = {
        "type": "object",
        "properties": {property_name: {"type": "string"}},
        "required": [property_name],
        "additionalProperties": False,
    }
    definition = _decision_model(Never, [_raw_tool(raw_schema)])

    schema = _prepare(definition).schema.to_dict()

    arguments = _resolve(_tool_call_item(schema)["properties"]["arguments"], schema)
    assert arguments == raw_schema


def _result_schema(schema: dict[str, Any]) -> dict[str, Any]:
    payload = _resolve(schema["properties"]["payload"], schema)
    return _resolve(payload["properties"]["result"], schema)


def test_mapping_result_is_lowered_and_decoded() -> None:
    definition = _decision_model(dict[str, str], [])

    schema = _prepare(definition).schema.to_dict()

    result_schema = _result_schema(schema)
    assert result_schema["type"] == "array"
    assert result_schema["items"]["additionalProperties"] is False
    assert result_schema["items"]["required"] == ["key", "value"]

    decision = _process(
        definition,
        {
            "payload": {
                "decision": "result",
                "result": [
                    {"key": "Maintainability", "value": "good"},
                    {"key": "Dependencies", "value": "current"},
                ],
            }
        },
    )

    assert isinstance(decision, ResultDecision)
    assert decision.result == {
        "Maintainability": "good",
        "Dependencies": "current",
    }


def test_mapping_values_can_be_mappings() -> None:
    definition = _decision_model(dict[str, dict[str, int]], [])

    result_schema = _result_schema(_prepare(definition).schema.to_dict())
    value_schema = result_schema["items"]["properties"]["value"]
    assert value_schema["type"] == "array"

    decision = _process(
        definition,
        {
            "payload": {
                "decision": "result",
                "result": [
                    {
                        "key": "outer",
                        "value": [{"key": "inner", "value": 1}],
                    }
                ],
            }
        },
    )

    assert isinstance(decision, ResultDecision)
    assert decision.result == {"outer": {"inner": 1}}


def test_mapping_values_can_be_nested_to_arbitrary_depth() -> None:
    definition = _decision_model(dict[str, dict[str, dict[str, int]]], [])

    decision = _process(
        definition,
        {
            "payload": {
                "decision": "result",
                "result": [
                    {
                        "key": "outer",
                        "value": [
                            {
                                "key": "middle",
                                "value": [{"key": "inner", "value": 1}],
                            }
                        ],
                    }
                ],
            }
        },
    )

    assert isinstance(decision, ResultDecision)
    assert decision.result == {"outer": {"middle": {"inner": 1}}}


def test_mapping_constraints_are_lowered_to_entry_constraints() -> None:
    output_type = Annotated[dict[str, str], Field(min_length=1, max_length=2)]
    definition = _decision_model(output_type, [])

    result_schema = _result_schema(_prepare(definition).schema.to_dict())

    assert result_schema["minItems"] == 1
    assert result_schema["maxItems"] == 2


class _HybridObject(BaseModel):
    model_config = ConfigDict(extra="allow")

    fixed: str
    __pydantic_extra__: dict[str, int]  # pyright: ignore[reportIncompatibleVariableOverride]


def test_hybrid_object_is_not_lowered_as_a_dictionary() -> None:
    definition = _decision_model(_HybridObject, [])

    with pytest.raises(
        ValueError,
        match="objects combining fixed properties with dictionary values",
    ):
        _prepare(definition)


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
            "payload": {
                "decision": "result",
                "result": {
                    "issues_by_perspective": [
                        {
                            "key": "Maintainability",
                            "value": [{"description": "Use clearer names."}],
                        }
                    ]
                },
            }
        },
    )

    assert isinstance(decision, ResultDecision)
    assert decision.result == _Report(
        issues_by_perspective={
            "Maintainability": [_Issue(description="Use clearer names.")]
        }
    )


def test_mapping_restoration_rejects_duplicate_keys() -> None:
    definition = _decision_model(dict[str, str], [])

    with pytest.raises(ValueError, match="duplicate mapping key"):
        _process(
            definition,
            {
                "payload": {
                    "decision": "result",
                    "result": [
                        {"key": "same", "value": "first"},
                        {"key": "same", "value": "second"},
                    ],
                }
            },
        )


def test_mapping_restoration_rejects_malformed_entries() -> None:
    definition = _decision_model(dict[str, str], [])

    with pytest.raises(ValueError, match="contain only key and value"):
        _process(
            definition,
            {
                "payload": {
                    "decision": "result",
                    "result": [{"key": "missing-value"}],
                }
            },
        )


def test_mapping_nested_in_list_is_lowered_and_decoded() -> None:
    definition = _decision_model(list[dict[str, int]], [])

    decision = _process(
        definition,
        {
            "payload": {
                "decision": "result",
                "result": [[{"key": "count", "value": 3}]],
            }
        },
    )

    assert isinstance(decision, ResultDecision)
    assert decision.result == [{"count": 3}]


@dataclass
class _MappedBranch:
    labels: dict[str, int]


@dataclass
class _TextBranch:
    text: str


def test_mapping_nested_in_union_is_lowered_and_decoded() -> None:
    definition = _decision_model(_MappedBranch | _TextBranch, [])

    decision = _process(
        definition,
        {
            "payload": {
                "decision": "result",
                "result": {
                    "labels": [{"key": "important", "value": 2}],
                },
            }
        },
    )

    assert isinstance(decision, ResultDecision)
    assert decision.result == _MappedBranch(labels={"important": 2})


@dataclass
class _DictionaryBranch:
    x: dict[str, int]


@dataclass
class _LaterDictionaryBranch:
    x: list[int]
    y: dict[str, int]


def test_mapping_union_selects_the_fully_valid_wire_branch() -> None:
    definition = _decision_model(_DictionaryBranch | _LaterDictionaryBranch, [])

    decision = _process(
        definition,
        {
            "payload": {
                "decision": "result",
                "result": {
                    "x": [1],
                    "y": [{"key": "n", "value": 2}],
                },
            }
        },
    )

    assert isinstance(decision, ResultDecision)
    assert decision.result == _LaterDictionaryBranch(x=[1], y={"n": 2})


async def _categorize(labels: dict[str, int]) -> None:
    pass


async def _categorize_nested(values: dict[str, dict[str, int]]) -> None:
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
            "payload": {
                "decision": "tool_calls",
                "tool_calls": [
                    {
                        "name": "categorize",
                        "arguments": {"labels": [{"key": "important", "value": 2}]},
                    }
                ],
            }
        },
        ToolCallIdRegistry(),
    )

    assert isinstance(decision, ToolCallsDecision)
    assert decision.calls[0].arguments == {"labels": {"important": 2}}


def test_nested_mapping_tool_argument_is_lowered_and_decoded() -> None:
    definition = _decision_model(
        Never,
        [_signature_tool(_categorize_nested, name="categorize_nested")],
    )

    decision = _process(
        definition,
        {
            "payload": {
                "decision": "tool_calls",
                "tool_calls": [
                    {
                        "name": "categorize_nested",
                        "arguments": {
                            "values": [
                                {
                                    "key": "outer",
                                    "value": [{"key": "inner", "value": 1}],
                                }
                            ]
                        },
                    }
                ],
            }
        },
        ToolCallIdRegistry(),
    )

    assert isinstance(decision, ToolCallsDecision)
    assert decision.calls[0].arguments == {"values": {"outer": {"inner": 1}}}


def test_step_decision_spec_rejects_tool_modes_without_tools() -> None:
    with pytest.raises(ValueError, match="require at least one tool"):
        StepDecisionSpec(
            "StepDecision",
            Never,
            [],
            StepDecisionMode.TOOLS_REQUIRED,
        )

    with pytest.raises(ValueError, match="require at least one tool"):
        StepDecisionSpec(
            "StepDecision",
            str,
            [],
            StepDecisionMode.TOOLS_OR_RESULT,
        )


class TestToolCallValidation:
    """The step-decision validator validates tool arguments end-to-end."""

    def _strategy(self, content: str) -> LLMInferenceStrategy:
        client = Mock(spec=LLMClient)
        client.complete.return_value = LLMResponse(
            content=content,
            structured_output=LLMOutput.from_json(
                cast(dict[str, Any], json.loads(content))["payload"]
            ),
        )
        renderer = Mock()
        renderer.render_instructions.return_value = "instructions"
        renderer.render_invocation.return_value = "invocation"
        renderer.render_response_feedback.return_value = "feedback"
        return LLMInferenceStrategy(
            llm_client=client,
            result_format_factory=PydanticModelBackend(),
            prompt_renderer=renderer,
        )

    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.add(ask_user, name="ask_user")
        return registry

    async def _decide(self, payload: dict[str, Any]) -> StepDecision:
        strategy = self._strategy(json.dumps({"payload": payload}))
        return await strategy.decide_next_step(
            FunctionInfo(
                qualname="test",
                instructions="chat",
                bound_arguments={},
                type_hints={},
                return_type=Never,
                args=(),
                kwargs={},
            ),
            [],
            self._registry(),
            _MockPublisher(),
        )

    async def test_rejects_unknown_tool_with_specific_cause(self) -> None:
        with pytest.raises(InvalidInferenceResponseError) as exc_info:
            await self._decide(
                {
                    "decision": "tool_calls",
                    "tool_calls": [{"name": "unknown", "arguments": {}}],
                }
            )

        assert isinstance(exc_info.value.__cause__, UnknownToolDecisionError)
        assert exc_info.value.__cause__.tool_name == "unknown"

    async def test_rejects_missing_required_argument(self) -> None:
        with pytest.raises(InvalidInferenceResponseError, match="question"):
            await self._decide(
                {
                    "decision": "tool_calls",
                    "tool_calls": [{"name": "ask_user", "arguments": {}}],
                }
            )

    async def test_rejects_empty_min_length_argument(self) -> None:
        with pytest.raises(InvalidInferenceResponseError, match="non-empty"):
            await self._decide(
                {
                    "decision": "tool_calls",
                    "tool_calls": [{"name": "ask_user", "arguments": {"question": ""}}],
                }
            )

    async def test_rejects_unknown_argument(self) -> None:
        with pytest.raises(InvalidInferenceResponseError, match="LLM output failed"):
            await self._decide(
                {
                    "decision": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "ask_user",
                            "arguments": {"question": "Hi", "extra": 1},
                        }
                    ],
                }
            )

    async def test_accepts_valid_arguments(self) -> None:
        decision = await self._decide(
            {
                "decision": "tool_calls",
                "tool_calls": [
                    {"name": "ask_user", "arguments": {"question": "Hello"}}
                ],
            }
        )

        assert isinstance(decision, ToolCallsDecision)
        assert decision.calls[0].name == "ask_user"
        assert decision.calls[0].arguments == {"question": "Hello"}
