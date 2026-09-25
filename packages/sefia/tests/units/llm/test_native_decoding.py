import pytest
from sefia.llm import JsonSnapshot, ToolCall
from sefia.llm.transports._native._decoding import decode_native_tool_calls


@pytest.mark.parametrize(
    "calls",
    [
        [],
        [ToolCall(id="provider-id", name="lookup", arguments=JsonSnapshot.capture([]))],
    ],
)
def test_native_decoding_rejects_missing_calls_or_nonobject_arguments(
    calls: list[ToolCall],
) -> None:
    with pytest.raises(ValueError):
        decode_native_tool_calls(calls, None)


def test_native_decoding_preserves_application_calls() -> None:
    calls = [
        ToolCall(
            id="provider-id",
            name="lookup",
            arguments=JsonSnapshot.capture({"key": "item"}),
        )
    ]
    assert decode_native_tool_calls(calls, None).to_json_compatible() == {
        "decision": "tool_calls",
        "tool_calls": [{"name": "lookup", "arguments": {"key": "item"}}],
    }
