from dataclasses import dataclass

from sefia.llm import (
    Message,
    ToolCall,
)
from sefia.llm.json_schema import JsonSchemaDocument
from sefia.llm.step_decision import DecisionSpec, StepTool, ToolSchemaSource
from sefia.llm.structured_data import StructuredData
from sefia.pydantic import PydanticModelBackend
from sefia_litellm._request import build_completion_request


@dataclass
class _CityResult:
    city: str


def _decision_spec() -> DecisionSpec:
    return DecisionSpec.for_inference(
        output_type=_CityResult,
        tools=[],
        result_format_factory=PydanticModelBackend(),
    )


def test_request_skips_structured_output_without_decision_model() -> None:
    request = build_completion_request(
        messages=[Message(role="user", content="Hello")],
        tools=None,
        decision_spec=None,
        client_kwargs={},
        stream=False,
    )

    call_args = request.api_kwargs
    assert "response_format" not in call_args
    assert request.messages == [{"role": "user", "content": "Hello"}]


def test_request_sends_correct_request_to_litellm():
    messages = [Message(role="user", content="Hello")]
    tools = [
        StepTool(
            name="get_weather",
            description="",
            arguments=JsonSchemaDocument.from_mapping(
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                }
            ),
            schema_source=ToolSchemaSource.GENERATED,
        )
    ]
    decision_spec = _decision_spec()

    request = build_completion_request(
        messages=messages,
        tools=tools,
        decision_spec=decision_spec,
        client_kwargs={"temperature": 0.5},
        stream=False,
    )

    call_args = request.api_kwargs
    assert request.messages == [{"role": "user", "content": "Hello"}]
    assert call_args["tools"][0]["function"] == {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }
    assert call_args["response_format"]["type"] == "json_schema"
    assert request.decision_format is not None
    assert (
        call_args["response_format"]["json_schema"]["schema"]
        == request.decision_format.schema.to_dict()
    )
    assert call_args["temperature"] == 0.5


def test_request_encodes_native_tool_call_history_for_wire_schema() -> None:
    tool = StepTool(
        name="categorize",
        description="",
        arguments=JsonSchemaDocument.from_mapping(
            {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    }
                },
                "required": ["labels"],
                "additionalProperties": False,
            }
        ),
        schema_source=ToolSchemaSource.GENERATED,
    )
    messages = [
        Message(role="user", content="Categorize."),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="categorize",
                    arguments=StructuredData.from_json({"labels": {"important": 2}}),
                )
            ],
        ),
        Message(role="tool", content="done", tool_call_id="call-1"),
    ]

    request = build_completion_request(
        messages=messages,
        tools=[tool],
        decision_spec=None,
        client_kwargs={},
        stream=False,
    )

    sent_call = request.messages[1]["tool_calls"][0]
    assert sent_call == {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "categorize",
            "arguments": ('{"labels":[{"key":"important","value":2}]}'),
        },
    }


def test_request_closes_user_tool_schema_without_mutating_original() -> None:
    raw_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    document = JsonSchemaDocument.from_mapping(raw_schema)
    tool = StepTool(
        name="search",
        description="",
        arguments=document,
        schema_source=ToolSchemaSource.USER_DEFINED,
    )

    request = build_completion_request(
        messages=[],
        tools=[tool],
        decision_spec=None,
        client_kwargs={},
        stream=False,
    )

    assert request.api_kwargs["tools"][0]["function"]["parameters"] == {
        **raw_schema,
        "additionalProperties": False,
    }
    assert document.to_dict() == raw_schema
