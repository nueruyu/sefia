from dataclasses import FrozenInstanceError, fields
from unittest.mock import Mock

import pytest
from sefia.llm import JsonSnapshot, LLMCompletion, Message, PromptRenderer, ToolCall
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.transports import (
    DecisionToolCalls,
    DecisionToolResult,
    RejectedDecision,
)
from sefia.llm.transports._messages import (
    build_decision_messages,
    build_text_decision_messages,
)
from sefia.pydantic import PydanticResultFormatFactory
from sefia.testing import make_decision_request


def _messages(rejected: RejectedDecision) -> list[Message]:
    decision_spec = DecisionSpec.for_inference(
        output_type=str,
        tools=[],
        result_format_factory=PydanticResultFormatFactory(),
    )
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "prompt"
    return build_decision_messages(
        request=make_decision_request(decision_spec, rejected=rejected),
        renderer=renderer,
        prompt_tools=(),
        history_messages=[],
        response_instructions="respond",
    )


def test_rejection_message_describes_empty_previous_response() -> None:
    message = _messages(
        RejectedDecision(completion=LLMCompletion(), reason="empty response")
    )[-2]

    assert message.role == "user"
    assert isinstance(message.content, str)
    assert "The previous response was empty." in message.content
    assert "Reason: empty response" in message.content
    assert "Return a corrected response" in message.content


def test_rejection_message_safely_fences_previous_content() -> None:
    message = _messages(
        RejectedDecision(
            completion=LLMCompletion(content="invalid ```json"),
            reason="invalid schema",
        )
    )[-2]

    assert isinstance(message.content, str)
    assert "````text\ninvalid ```json\n````" in message.content
    assert "Reason: invalid schema" in message.content


def test_rejection_message_represents_structured_completion_output() -> None:
    completion = LLMCompletion(
        tool_calls=[
            ToolCall(
                id="call-1",
                name="lookup",
                arguments=JsonSnapshot.capture({"query": "lost"}),
            )
        ],
        structured_output=JsonSnapshot.capture({"decision": "invalid"}),
    )

    message = _messages(
        RejectedDecision(completion=completion, reason="invalid decision")
    )[-2]

    assert isinstance(message.content, str)
    assert (
        '{"tool_calls":[{"id":"call-1","name":"lookup",'
        '"arguments":{"query":"lost"}}],'
        '"structured_output":{"decision":"invalid"}}' in message.content
    )
    assert "Reason: invalid decision" in message.content


def test_build_decision_messages_owns_final_framing_order() -> None:
    before = Message(role="developer", content="before")
    after = Message(role="user", content="after")
    history = Message(role="assistant", content="history")
    rejected = RejectedDecision(
        completion=LLMCompletion(content="rejected"),
        reason="invalid",
    )
    decision_spec = DecisionSpec.for_inference(
        output_type=str,
        tools=[],
        result_format_factory=PydanticResultFormatFactory(),
    )
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "inference"

    messages = build_decision_messages(
        request=make_decision_request(
            decision_spec,
            messages_before=(before,),
            messages_after=(after,),
            rejected=rejected,
        ),
        renderer=renderer,
        prompt_tools=(),
        history_messages=[history],
        response_instructions="respond",
    )

    assert [message.content for message in messages[:4]] == [
        "before",
        "inference",
        "after",
        "history",
    ]
    repair = messages[-2].content
    assert isinstance(repair, str)
    assert "Correct the previous response" in repair
    assert messages[-1].content == "## Response\n\nrespond"
    assert messages[0] is before
    assert messages[2] is after
    assert messages[3] is history


def test_default_first_step_constructs_one_combined_message() -> None:
    decision_spec = DecisionSpec.for_inference(
        output_type=str,
        tools=[],
        result_format_factory=PydanticResultFormatFactory(),
    )
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "inference"

    messages = build_decision_messages(
        request=make_decision_request(decision_spec),
        renderer=renderer,
        prompt_tools=(),
        history_messages=[],
        response_instructions="respond",
    )

    assert messages == [
        Message(role="user", content="inference\n\n## Response\n\nrespond")
    ]


def test_build_text_decision_messages_owns_text_history_representation() -> None:
    decision_spec = DecisionSpec.for_inference(
        output_type=str,
        tools=[],
        result_format_factory=PydanticResultFormatFactory(),
    )
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "prompt"
    request = make_decision_request(
        decision_spec,
        history=(
            DecisionToolCalls(
                calls=(
                    ToolCall(
                        id="call-1",
                        name="lookup",
                        arguments=JsonSnapshot.capture({"query": "sefia"}),
                    ),
                )
            ),
            DecisionToolResult(
                tool_call_id="call-1",
                result=JsonSnapshot.capture({"value": "found"}),
            ),
        ),
    )

    messages = build_text_decision_messages(
        request=request,
        renderer=renderer,
        response_instructions="respond",
    )

    assert [message.role for message in messages] == ["user", "user", "user"]
    content = messages[1].content
    assert isinstance(content, str)
    assert "Previous tool interactions" in content
    assert '"arguments": {' in content
    assert '"query": "sefia"' in content
    assert '"result": {' in content
    assert '"value": "found"' in content


def test_decision_request_is_frozen_and_adds_rejection_as_a_new_value() -> None:
    decision_spec = DecisionSpec.for_inference(
        output_type=str,
        tools=[],
        result_format_factory=PydanticResultFormatFactory(),
    )
    request = make_decision_request(decision_spec)
    rejected = RejectedDecision(
        completion=LLMCompletion(content="invalid"),
        reason="invalid decision",
    )

    retry = request.with_rejection(rejected)

    assert request.rejected is None
    assert retry.rejected is rejected
    assert retry.function is request.function
    assert retry.arguments is request.arguments
    assert retry.messages_before is request.messages_before
    assert retry.messages_after is request.messages_after
    assert retry.history is request.history
    assert {field.name for field in fields(request)} == {
        "messages_before",
        "function",
        "arguments",
        "messages_after",
        "decision_spec",
        "history",
        "rejected",
    }
    with pytest.raises(FrozenInstanceError):
        setattr(request, "rejected", rejected)
