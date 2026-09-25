from dataclasses import FrozenInstanceError

import pytest

from sefia.llm import Message, ToolCall
from sefia.llm.structured_data import StructuredData


class _MutableValue:
    def __init__(self, value: str) -> None:
        self.value = value


def test_message_fields_are_immutable() -> None:
    message = Message(role="user", content="hello")

    with pytest.raises(FrozenInstanceError):
        setattr(message, "role", "assistant")


def test_message_owns_nested_content_input() -> None:
    source = [{"type": "text", "text": "original", "metadata": {"index": 1}}]
    message = Message(role="developer", content=source)

    source[0]["text"] = "changed"
    metadata = source[0]["metadata"]
    assert isinstance(metadata, dict)
    metadata["index"] = 2

    assert message.content == [
        {"type": "text", "text": "original", "metadata": {"index": 1}}
    ]


def test_message_content_returns_a_detached_nested_value() -> None:
    message = Message(
        role="developer",
        content=[{"type": "text", "text": "original", "metadata": {"index": 1}}],
    )

    content = message.content
    assert isinstance(content, list)
    content[0]["text"] = "changed"
    metadata = content[0]["metadata"]
    assert isinstance(metadata, dict)
    metadata["index"] = 2

    assert message.content == [
        {"type": "text", "text": "original", "metadata": {"index": 1}}
    ]


def test_message_owns_mutable_content_leaf_input() -> None:
    source = _MutableValue("original")
    message = Message(role="user", content=[source])

    source.value = "changed"

    content = message.content
    assert isinstance(content, list)
    assert isinstance(content[0], _MutableValue)
    assert content[0].value == "original"


def test_message_returns_a_detached_mutable_content_leaf() -> None:
    message = Message(role="user", content=[_MutableValue("original")])

    content = message.content
    assert isinstance(content, list)
    assert isinstance(content[0], _MutableValue)
    content[0].value = "changed"

    projected = message.content
    assert isinstance(projected, list)
    assert isinstance(projected[0], _MutableValue)
    assert projected[0].value == "original"


def test_message_owns_tool_call_collection() -> None:
    call = ToolCall(
        id="call-1",
        name="lookup",
        arguments=StructuredData.from_json({"query": "sefia"}),
    )
    source = [call]
    message = Message(role="assistant", tool_calls=source)

    source.clear()

    assert message.tool_calls == (call,)


def test_tool_call_is_immutable_and_has_structural_equality() -> None:
    first = ToolCall(
        id="call-1",
        name="lookup",
        arguments=StructuredData.from_json({"query": "sefia"}),
    )
    second = ToolCall(
        id="call-1",
        name="lookup",
        arguments=StructuredData.from_json({"query": "sefia"}),
    )

    assert first == second
    with pytest.raises(FrozenInstanceError):
        setattr(first, "name", "changed")


def test_message_has_structural_equality() -> None:
    assert Message(role="user", content=[{"text": "hello"}]) == Message(
        role="user", content=[{"text": "hello"}]
    )
