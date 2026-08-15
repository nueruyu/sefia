import json
from dataclasses import dataclass, make_dataclass
from typing import Annotated, Any, Literal, Never, cast
from unittest.mock import Mock

import pytest
from pydantic import Field, TypeAdapter, ValidationError

from sefia._tool_system import (
    JsonSchemaToolEntry,
    SignatureToolEntry,
    ToolEntry,
    ToolRegistry,
)
from sefia.event_system import Event, EventPublisher
from sefia.exceptions import InvalidInferenceResponseError, UnknownToolDecisionError
from sefia.inference import FunctionInfo, InferenceDecision, ToolCallDecision
from sefia.inference import ResultDecision
from sefia.llm import LLMClient, LLMInferenceStrategy, LLMResponse
from sefia.llm.decision import DecisionModelSpec
from sefia.llm._execution_directors import (
    OutputOnlyDirector,
    ToolEnabledDirector,
    ToolOnlyDirector,
)
from sefia.llm._tool_call_ids import ToolCallIdRegistry
from sefia.llm._arg_stream import ToolArgStreamer
from sefia.pydantic import PydanticModelBackend
from sefia.pydantic._decision_model import _unknown_tool_name_from_error
from sefia.streaming import ArgStream, StringEnd
from sefia_litellm._schema import LiteLLMSchemaAdapter


def _prepare(director: Any):
    return LiteLLMSchemaAdapter().build(director.build_decision_schema())


def _process(director: Any, data: Any, tool_call_ids: ToolCallIdRegistry | None = None):
    prepared = _prepare(director)
    return director.process_response_data(prepared.decode(data), tool_call_ids)


def test_prepared_schema_removes_payload_from_stream_paths() -> None:
    prepared = _prepare(OutputOnlyDirector(PydanticModelBackend(), str, []))

    assert prepared.normalize_stream_path(
        ("payload", "tool_calls", 0, "arguments", "question")
    ) == ("tool_calls", 0, "arguments", "question")


async def test_payload_stream_reaches_preview_as_a_logical_argument() -> None:
    prepared = _prepare(ToolOnlyDirector(PydanticModelBackend(), Never, [_tool()]))
    events: list[object] = []

    async def collect(_tool_call_id: str, stream: ArgStream) -> None:
        async for event in stream:
            events.append(event)

    streamer = ToolArgStreamer(
        {"ask_user": collect},
        lambda index: f"call-{index}",
        prepared.normalize_stream_path,
    )
    streamer.on_token(
        '{"payload":{"decision":"tool_calls","tool_calls":['
        '{"name":"ask_user","arguments":{"question":"Hello"}}]}}'
    )
    await streamer.close()

    assert StringEnd(name="question", value="Hello") in events


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


def _raw_tool(schema: dict[str, Any]) -> ToolEntry:
    async def handler(**kwargs: Any) -> str:
        return str(kwargs)

    return JsonSchemaToolEntry(handler, name="raw_tool", parameters=schema)


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follow a top-level ``$ref`` into ``$defs`` so assertions can inspect the
    embedded per-tool schemas regardless of how Pydantic hoists definitions."""
    if "$ref" in schema:
        key = schema["$ref"].split("/")[-1]
        return root["$defs"][key]
    return schema


def _tool_calls_array(schema: dict[str, Any]) -> dict[str, Any]:
    payload = _resolve(schema["properties"]["payload"], schema)
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
    director = ToolOnlyDirector(PydanticModelBackend(), Never, [_tool()])

    schema = _prepare(director).schema

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
    director = ToolOnlyDirector(
        PydanticModelBackend(), Never, [_signature_tool(_research, name="research")]
    )

    schema = _prepare(director).schema

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

    director = ToolEnabledDirector(PydanticModelBackend(), Output, [_tool()])

    schema = director.build_decision_schema().schema

    assert schema["$defs"]["Output"]["properties"]["arguments"] == {
        "additionalProperties": {"type": "integer"},
        "title": "Arguments",
        "type": "object",
    }


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
    director = ToolOnlyDirector(PydanticModelBackend(), Never, [_raw_tool(raw_schema)])

    schema = _prepare(director).schema
    arguments = _resolve(_tool_call_item(schema)["properties"]["arguments"], schema)

    assert "$defs" not in arguments
    assert arguments["properties"]["item"]["$ref"] == "#/$defs/Item"
    assert schema["$defs"]["Item"]["properties"]["name"]["type"] == "string"


def test_conflicting_tool_definition_names_are_renamed() -> None:
    first_type = make_dataclass("Shared", [("text", str)])
    second_type = make_dataclass("Shared", [("count", int)])

    async def first(value: Any) -> None:
        pass

    async def second(value: Any) -> None:
        pass

    first.__annotations__["value"] = first_type
    second.__annotations__["value"] = second_type
    director = ToolOnlyDirector(
        PydanticModelBackend(),
        Never,
        [
            _signature_tool(first, name="first"),
            _signature_tool(second, name="second"),
        ],
    )

    schema = _prepare(director).schema

    shared_definitions = {
        name: definition
        for name, definition in schema["$defs"].items()
        if name == "Shared" or name.startswith("second__Shared")
    }
    assert len(shared_definitions) == 2
    assert {
        tuple(definition["properties"]) for definition in shared_definitions.values()
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
        assert reference.removeprefix("#/$defs/") in root["$defs"]
    for value in node_dict.values():
        _assert_local_references_resolve(value, root)


def test_compatible_raw_tool_schema_is_preserved_verbatim() -> None:
    raw_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    director = ToolOnlyDirector(PydanticModelBackend(), Never, [_raw_tool(raw_schema)])

    schema = _prepare(director).schema

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
    director = ToolOnlyDirector(PydanticModelBackend(), Never, [_raw_tool(raw_schema)])

    with pytest.raises(ValueError, match=message):
        _prepare(director)


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
    director = ToolOnlyDirector(PydanticModelBackend(), Never, [_raw_tool(raw_schema)])

    with pytest.raises(ValueError, match=rf"{keyword} is not supported"):
        _prepare(director)


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
    director = ToolOnlyDirector(PydanticModelBackend(), Never, [_raw_tool(raw_schema)])

    schema = _prepare(director).schema

    arguments = _resolve(_tool_call_item(schema)["properties"]["arguments"], schema)
    assert arguments == raw_schema


def _result_schema(schema: dict[str, Any]) -> dict[str, Any]:
    payload = _resolve(schema["properties"]["payload"], schema)
    return _resolve(payload["properties"]["result"], schema)


def test_mapping_result_is_lowered_and_decoded() -> None:
    director = OutputOnlyDirector(PydanticModelBackend(), dict[str, str], [])

    schema = _prepare(director).schema

    result_schema = _result_schema(schema)
    assert result_schema["type"] == "array"
    assert result_schema["items"]["additionalProperties"] is False
    assert result_schema["items"]["required"] == ["key", "value"]

    decision = _process(
        director,
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


def test_mapping_constraints_are_lowered_to_entry_constraints() -> None:
    output_type = Annotated[dict[str, str], Field(min_length=1, max_length=2)]
    director = OutputOnlyDirector(PydanticModelBackend(), output_type, [])

    result_schema = _result_schema(_prepare(director).schema)

    assert result_schema["minItems"] == 1
    assert result_schema["maxItems"] == 2


@dataclass
class _Issue:
    description: str


@dataclass
class _Report:
    issues_by_perspective: dict[str, list[_Issue]]


def test_nested_mapping_result_is_lowered_and_decoded() -> None:
    director = OutputOnlyDirector(PydanticModelBackend(), _Report, [])

    schema = _prepare(director).schema
    report_schema = _result_schema(schema)
    report_schema = _resolve(report_schema, schema)
    mapping_schema = report_schema["properties"]["issues_by_perspective"]
    assert mapping_schema["type"] == "array"

    decision = _process(
        director,
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


def test_mapping_decoder_rejects_duplicate_keys() -> None:
    director = OutputOnlyDirector(PydanticModelBackend(), dict[str, str], [])

    with pytest.raises(ValueError, match="duplicate mapping key"):
        _process(
            director,
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


def test_mapping_decoder_rejects_malformed_entries() -> None:
    director = OutputOnlyDirector(PydanticModelBackend(), dict[str, str], [])

    with pytest.raises(ValueError, match="contain only key and value"):
        _process(
            director,
            {
                "payload": {
                    "decision": "result",
                    "result": [{"key": "missing-value"}],
                }
            },
        )


def test_mapping_nested_in_list_is_lowered_and_decoded() -> None:
    director = OutputOnlyDirector(PydanticModelBackend(), list[dict[str, int]], [])

    decision = _process(
        director,
        {
            "payload": {
                "decision": "result",
                "result": [[{"key": "count", "value": 3}]],
            }
        },
    )

    assert isinstance(decision, ResultDecision)
    assert decision.result == [{"count": 3}]


async def _categorize(labels: dict[str, int]) -> None:
    pass


def test_mapping_tool_argument_is_lowered_and_decoded() -> None:
    director = ToolOnlyDirector(
        PydanticModelBackend(),
        Never,
        [_signature_tool(_categorize, name="categorize")],
    )

    schema = _prepare(director).schema
    arguments = _resolve(_tool_call_item(schema)["properties"]["arguments"], schema)
    assert arguments["properties"]["labels"]["type"] == "array"

    decision = _process(
        director,
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

    assert isinstance(decision, ToolCallDecision)
    assert decision.calls[0].arguments == {"labels": {"important": 2}}


def test_unknown_tool_name_ignores_root_literal_errors() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TypeAdapter(Literal["expected"]).validate_python("actual")

    assert _unknown_tool_name_from_error(exc_info.value) is None


def test_decision_model_spec_rejects_tool_modes_without_tools() -> None:
    with pytest.raises(ValueError, match="require at least one tool"):
        DecisionModelSpec.tool_only(
            name="LLMDecision",
            output_type=Never,
            tools=[],
        )

    with pytest.raises(ValueError, match="require at least one tool"):
        DecisionModelSpec.tool_enabled(
            name="LLMDecision",
            output_type=str,
            tools=[],
        )


class TestToolCallValidation:
    """The decision model validates tool arguments end-to-end via the backend."""

    def _strategy(self, content: str) -> LLMInferenceStrategy:
        client = Mock(spec=LLMClient)
        client.complete.return_value = LLMResponse(content=content)
        client.prepare_output_schema.side_effect = LiteLLMSchemaAdapter().build
        formatter = Mock()
        formatter.format_arguments.return_value = "<arguments/>"
        return LLMInferenceStrategy(
            llm_client=client,
            decision_builder=PydanticModelBackend(),
            prompt_formatter=formatter,
        )

    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.add(ask_user, name="ask_user")
        return registry

    async def _decide(self, payload: dict[str, Any]) -> InferenceDecision:
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

        assert isinstance(decision, ToolCallDecision)
        assert decision.calls[0].name == "ask_user"
        assert decision.calls[0].arguments == {"question": "Hello"}
