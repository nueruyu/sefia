from collections.abc import Callable
from unittest.mock import AsyncMock

from litellm import ChatCompletionMessageToolCall, ModelResponse
from sefia._tool_system import ToolRegistry
from sefia.json_schema import JsonSchemaDocument
from sefia.llm import (
    Message,
)
from sefia.llm.step_decision import DecisionSpec, StepTool, ToolSchemaSource
from sefia.pydantic import PydanticResultFormatFactory
from sefia_litellm._client import (
    LiteLLMClient,
)

_ResponseFactory = Callable[..., ModelResponse]


async def test_complete_envelopes_tool_or_result_union(
    mock_acompletion: AsyncMock,
    make_litellm_response: _ResponseFactory,
) -> None:
    def lookup(key: str) -> str:
        return key

    registry = ToolRegistry()
    registry.add(lookup, name="lookup")
    decision_spec = DecisionSpec.for_inference(
        output_type=str,
        tools=registry.get_all(),
        result_format_factory=PydanticResultFormatFactory(),
    )
    mock_acompletion.return_value = make_litellm_response(
        content=(
            '{"payload":{"decision":"tool_calls","tool_calls":['
            '{"name":"lookup","arguments":{"key":"item"}}]}}'
        )
    )

    response = await LiteLLMClient(model="gpt-4o").complete(
        [Message(role="user", content="Look up the item.")],
        decision_spec=decision_spec,
    )

    schema = mock_acompletion.call_args.kwargs["response_format"]["json_schema"][
        "schema"
    ]
    assert schema["type"] == "object"
    assert "anyOf" not in schema
    assert "anyOf" in schema["properties"]["payload"]
    assert response.structured_output is not None
    assert response.structured_output.to_json_compatible() == {
        "decision": "tool_calls",
        "tool_calls": [{"name": "lookup", "arguments": {"key": "item"}}],
    }


async def test_complete_translates_native_tool_schema_and_arguments(
    mock_acompletion: AsyncMock,
    make_litellm_response: _ResponseFactory,
) -> None:
    tool = StepTool(
        name="categorize",
        description="Categorize labels.",
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
    mock_acompletion.return_value = make_litellm_response(
        finish_reason="tool_calls",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call-1",
                function={
                    "name": "categorize",
                    "arguments": ('{"labels":[{"key":"important","value":2}]}'),
                },
                type="function",
            )
        ],
    )

    response = await LiteLLMClient(model="gpt-4o").complete(
        [Message(role="user", content="Categorize.")],
        tools=[tool],
    )

    sent_schema = mock_acompletion.call_args.kwargs["tools"][0]["function"][
        "parameters"
    ]
    assert sent_schema["properties"]["labels"]["type"] == "array"
    assert response.tool_calls[0].arguments.to_json_compatible() == {
        "labels": {"important": 2}
    }
