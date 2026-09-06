import pytest
from sefia.llm import ToolCall
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports._native._decoding import decode_native_tool_calls


@pytest.mark.parametrize(
    "calls",
    [
        [],
        [
            ToolCall(
                id="provider-id", name="lookup", arguments=StructuredData.from_json([])
            )
        ],
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
            arguments=StructuredData.from_json({"key": "item"}),
        )
    ]
    assert decode_native_tool_calls(calls, None).tree == {
        "decision": "tool_calls",
        "tool_calls": [{"name": "lookup", "arguments": {"key": "item"}}],
    }
