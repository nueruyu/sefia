from typing import Annotated, Any, Never

import jsonschema
import pytest
from pydantic import Field
from sefia import JsonSchemaToolEntry, ToolRegistry
from sefia.exceptions import UnknownToolDecisionError
from sefia.inference import ToolCallsDecision
from sefia.llm._tool_call_ids import ToolCallIdRegistry
from sefia.llm.json_schema import JsonValue
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.structured_data import StructuredData
from sefia.pydantic import PydanticModelBackend


def _noop(**kwargs: Any) -> None:
    pass


def test_a_malformed_schema_is_rejected_up_front():
    tool = JsonSchemaToolEntry(
        _noop,
        name="invalid",
        parameters={"type": "not-a-type"},
    )
    with pytest.raises(jsonschema.SchemaError):
        DecisionSpec.for_inference(
            output_type=Never,
            tools=[tool],
            result_format_factory=PydanticModelBackend(),
        )


def test_a_schema_is_validated_under_its_declared_dialect():
    # Array-form ``items`` (tuple validation) is draft-07; under the default
    # 2020-12 dialect it would be rejected as malformed.
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "pair": {"items": [{"type": "string"}, {"type": "integer"}]},
        },
        "required": ["pair"],
    }

    tool = JsonSchemaToolEntry(_noop, name="pair", parameters=schema)
    decision_spec = DecisionSpec.for_inference(
        output_type=Never,
        tools=[tool],
        result_format_factory=PydanticModelBackend(),
    )
    tool_call_ids = ToolCallIdRegistry()

    valid = decision_spec.validate(
        StructuredData.from_json(
            {
                "decision": "tool_calls",
                "tool_calls": [{"name": "pair", "arguments": {"pair": ["a", 1]}}],
            }
        ),
        tool_call_ids,
    )
    assert isinstance(valid, ToolCallsDecision)
    assert valid.calls[0].arguments == {"pair": ["a", 1]}

    with pytest.raises(ValueError, match="Step decision validation failed"):
        decision_spec.validate(
            StructuredData.from_json(
                {
                    "decision": "tool_calls",
                    "tool_calls": [{"name": "pair", "arguments": {"pair": [1, "a"]}}],
                }
            ),
            tool_call_ids,
        )


def ask_user(question: Annotated[str, Field(min_length=1)]) -> str:
    return question


@pytest.fixture
def decision_spec() -> DecisionSpec:
    registry = ToolRegistry()
    registry.add(ask_user, name="ask_user")
    return DecisionSpec.for_inference(
        output_type=Never,
        tools=registry.get_all(),
        result_format_factory=PydanticModelBackend(),
    )


@pytest.mark.parametrize(
    "arguments", [{}, {"question": ""}, {"question": "Hi", "extra": 1}]
)
def test_rejects_invalid_tool_arguments(
    decision_spec: DecisionSpec, arguments: JsonValue
) -> None:
    data = StructuredData.from_json(
        {
            "decision": "tool_calls",
            "tool_calls": [{"name": "ask_user", "arguments": arguments}],
        }
    )
    with pytest.raises(ValueError):
        decision_spec.validate(data, ToolCallIdRegistry())


def test_accepts_valid_tool_arguments(decision_spec: DecisionSpec) -> None:
    data = StructuredData.from_json(
        {
            "decision": "tool_calls",
            "tool_calls": [{"name": "ask_user", "arguments": {"question": "Hello"}}],
        }
    )
    decision = decision_spec.validate(data, ToolCallIdRegistry())
    assert isinstance(decision, ToolCallsDecision)
    assert decision.calls[0].name == "ask_user"
    assert decision.calls[0].arguments == {"question": "Hello"}


def test_rejects_unknown_tool(decision_spec: DecisionSpec) -> None:
    data = StructuredData.from_json(
        {"decision": "tool_calls", "tool_calls": [{"name": "unknown", "arguments": {}}]}
    )
    with pytest.raises(UnknownToolDecisionError) as exc_info:
        decision_spec.validate(data, ToolCallIdRegistry())
    assert exc_info.value.tool_name == "unknown"


def test_never_mode_rejects_result(decision_spec: DecisionSpec) -> None:
    with pytest.raises(ValueError, match="Step decision validation failed"):
        decision_spec.validate(
            StructuredData.from_json({"decision": "result", "result": "bye"}), None
        )
