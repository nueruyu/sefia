from dataclasses import FrozenInstanceError, fields

import pytest
from sefia import DecisionContext
from sefia.inference import ToolCallResult
from sefia.llm import JsonSnapshot, LLMCompletion, Message, ToolCall
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.streaming import OutputStreamEvent
from sefia.llm.streaming import StringDelta as OutputStringDelta
from sefia.llm.transports import DecisionToolResult
from sefia.pydantic import PydanticResultFormatFactory
from sefia.testing import (
    LLMClientCase,
    MockLLMClient,
    ScriptedCompletion,
    StreamContent,
    StreamOutput,
    StreamReasoning,
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


async def test_mock_llm_client_replays_callbacks_in_event_order() -> None:
    completion = LLMCompletion(content="done")
    output_event = OutputStringDelta(("result",), "d")
    client = MockLLMClient(
        [
            ScriptedCompletion(
                completion,
                events=(
                    StreamReasoning("thinking"),
                    StreamContent("do"),
                    StreamOutput(output_event),
                    StreamContent("ne"),
                ),
            )
        ]
    )
    observed: list[tuple[str, str | OutputStreamEvent]] = []

    async def on_content(text: str) -> None:
        observed.append(("content", text))

    async def on_reasoning(text: str) -> None:
        observed.append(("reasoning", text))

    async def on_output(event: OutputStreamEvent) -> None:
        observed.append(("output", event))

    returned = await client.complete(
        [Message(role="user", content="hello")],
        stream_callback=on_content,
        reasoning_callback=on_reasoning,
        output_callback=on_output,
    )

    assert returned is completion
    assert observed == [
        ("reasoning", "thinking"),
        ("content", "do"),
        ("output", output_event),
        ("content", "ne"),
    ]


async def test_mock_llm_client_ignores_events_without_matching_callbacks() -> None:
    completion = LLMCompletion(content="done")
    client = MockLLMClient(
        [
            ScriptedCompletion(
                completion,
                events=(
                    StreamReasoning("thinking"),
                    StreamContent("done"),
                    StreamOutput(OutputStringDelta(("result",), "d")),
                ),
            )
        ]
    )
    content_chunks: list[str] = []

    async def on_content(text: str) -> None:
        content_chunks.append(text)

    returned = await client.complete(
        [Message(role="user", content="hello")],
        stream_callback=on_content,
    )

    assert returned is completion
    assert content_chunks == ["done"]


def test_scripted_completion_snapshots_event_sequence() -> None:
    events = [StreamContent("first")]
    scripted = ScriptedCompletion(LLMCompletion(), events=events)
    events.append(StreamContent("second"))

    assert scripted.events == (StreamContent("first"),)


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
