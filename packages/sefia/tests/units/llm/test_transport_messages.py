from unittest.mock import Mock

from sefia.llm import LLMCompletion, ToolCall
from sefia.llm import Message, PromptRenderer
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import (
    DecisionToolCalls,
    DecisionToolResult,
    RejectedDecision,
)
from sefia.llm.transports._messages import build_decision_messages
from sefia.llm.transports._text_protocol import text_history_messages
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
                arguments=StructuredData.from_json({"query": "lost"}),
            )
        ],
        structured_output=StructuredData.from_json({"decision": "invalid"}),
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
    assert messages[0] is not before
    assert messages[2] is not after


def test_text_history_messages_preserve_json_representation() -> None:
    messages = text_history_messages(
        (
            DecisionToolCalls(
                calls=(
                    ToolCall(
                        id="call-1",
                        name="lookup",
                        arguments=StructuredData.from_json({"query": "sefia"}),
                    ),
                )
            ),
            DecisionToolResult(
                tool_call_id="call-1",
                result=StructuredData.from_json({"value": "found"}),
            ),
        )
    )

    assert len(messages) == 1
    assert messages[0].role == "user"
    content = messages[0].content
    assert isinstance(content, str)
    assert '"arguments": {' in content
    assert '"query": "sefia"' in content
    assert '"result": {' in content
    assert '"value": "found"' in content
