from sefia.llm.transports import RejectedDecision
from sefia.llm.transports._messages import rejection_message


def test_rejection_message_describes_empty_previous_response() -> None:
    message = rejection_message(RejectedDecision(content=None, reason="empty response"))

    assert message.role == "user"
    assert isinstance(message.content, str)
    assert "The previous response was empty." in message.content
    assert "Reason: empty response" in message.content
    assert "Return a corrected response" in message.content


def test_rejection_message_safely_fences_previous_content() -> None:
    message = rejection_message(
        RejectedDecision(content="invalid ```json", reason="invalid schema")
    )

    assert isinstance(message.content, str)
    assert "````text\ninvalid ```json\n````" in message.content
    assert "Reason: invalid schema" in message.content
