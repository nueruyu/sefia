from types import SimpleNamespace

from sefia_litellm._native_tool_stream import (
    extract_native_tool_call_fragments,
)


def test_extracts_fragments_without_decoding_argument_json() -> None:
    fragments = extract_native_tool_call_fragments(
        [
            SimpleNamespace(
                index=2,
                function=SimpleNamespace(name="lookup", arguments='{"key":"'),
            )
        ]
    )

    assert len(fragments) == 1
    assert fragments[0].index == 2
    assert fragments[0].name == "lookup"
    assert fragments[0].arguments_json == '{"key":"'
