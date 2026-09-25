from dataclasses import FrozenInstanceError, fields

import pytest
from sefia import DecisionContext
from sefia.inference import ToolCallResult
from sefia.llm import JsonSnapshot, LLMCompletion, Message, ToolCall
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.transports import DecisionToolResult
from sefia.pydantic import PydanticResultFormatFactory
from sefia.testing import (
    LLMClientCase,
    MockLLMClient,
    make_decision_context,
    make_decision_request,
    make_function_info,
    make_step_context,
    make_tool_call_request,
)


def test_llm_client_cases_have_independent_default_messages() -> None:
    first = LLMClientCase(MockLLMClient([]), LLMCompletion())
    second = LLMClientCase(MockLLMClient([]), LLMCompletion())

    assert first.messages is not second.messages
    assert first.messages[0] == Message(role="user", content="Hello")
    assert second.messages[0].content == "Hello"


def test_test_data_factories_supply_independent_defaults() -> None:
    first_function = make_function_info()
    second_function = make_function_info()
    first_call = make_tool_call_request()
    second_call = make_tool_call_request()
    first_context = make_step_context()
    second_context = make_step_context()

    first_function.bound_arguments["changed"] = True
    first_call.arguments["changed"] = True
    first_context.history.extend([ToolCallResult(tool_call_id="call-1", result="done")])

    assert second_function.bound_arguments == {}
    assert second_call.arguments == {}
    assert second_context.history.items == ()


def test_test_data_factories_preserve_explicit_values() -> None:
    function = make_function_info(
        qualname="Agent.answer",
        instructions="Answer the question.",
        bound_arguments={"question": "Why?"},
        return_type=int,
    )
    call = make_tool_call_request(
        id="lookup-1",
        name="lookup",
        arguments={"key": "answer"},
    )
    history = (
        DecisionToolResult(
            tool_call_id=call.id,
            result=JsonSnapshot.from_scalar("found"),
        ),
    )
    decision_spec = DecisionSpec.for_inference(
        output_type=str,
        tools=[],
        result_format_factory=PydanticResultFormatFactory(),
    )

    request = make_decision_request(
        decision_spec,
        function=function,
        history=history,
    )

    assert request.function is function
    assert request.history == history
    assert request.function.bound_arguments == {"question": "Why?"}
    assert call.name == "lookup"


async def test_mock_llm_client_snapshots_core_messages() -> None:
    client = MockLLMClient([LLMCompletion(content="done")])

    await client.complete(
        [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="lookup",
                        arguments=JsonSnapshot.capture({"key": "item", "filter": None}),
                    )
                ],
            )
        ]
    )

    assert client.requests[0]["messages"] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "lookup",
                    "arguments": {"key": "item", "filter": None},
                }
            ],
        }
    ]


def test_decision_context_factory_defaults() -> None:
    assert make_decision_context() == DecisionContext(step=0)


def test_decision_context_only_exposes_the_step_and_is_frozen() -> None:
    ctx = make_decision_context(step=3)

    assert ctx.step == 3
    assert {field.name for field in fields(ctx)} == {"step"}
    with pytest.raises(FrozenInstanceError):
        setattr(ctx, "step", 4)
