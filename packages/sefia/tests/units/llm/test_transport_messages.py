from sefia.llm import LLMCompletion, ToolCall
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import RejectedDecision
from sefia.llm.transports._messages import _rejection_message


def test_rejection_message_describes_empty_previous_response() -> None:
    message = _rejection_message(
        RejectedDecision(completion=LLMCompletion(), reason="empty response")
    )

    assert message.role == "user"
    assert isinstance(message.content, str)
    assert "The previous response was empty." in message.content
    assert "Reason: empty response" in message.content
    assert "Return a corrected response" in message.content


def test_rejection_message_safely_fences_previous_content() -> None:
    message = _rejection_message(
        RejectedDecision(
            completion=LLMCompletion(content="invalid ```json"),
            reason="invalid schema",
        )
    )

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

    message = _rejection_message(
        RejectedDecision(completion=completion, reason="invalid decision")
    )

    assert isinstance(message.content, str)
    assert (
        '{"tool_calls":[{"id":"call-1","name":"lookup",'
        '"arguments":{"query":"lost"}}],'
        '"structured_output":{"decision":"invalid"}}' in message.content
    )
    assert "Reason: invalid decision" in message.content
