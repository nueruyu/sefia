from dataclasses import dataclass

from sefia.llm.streaming import StringEnd

from sefia_litellm._native_tool_stream import (
    NativeToolCallDelta,
    NativeToolCallStreamDecoder,
)


@dataclass
class _FunctionDelta:
    name: str | None
    arguments: str


@dataclass
class _CallDelta:
    index: int
    function: _FunctionDelta


def test_decodes_typed_tool_call_fragments() -> None:
    decoder = NativeToolCallStreamDecoder({})
    calls: list[NativeToolCallDelta] = [
        _CallDelta(
            index=2,
            function=_FunctionDelta(name="lookup", arguments='{"key":"'),
        )
    ]

    events = decoder.feed(calls)

    assert StringEnd(("tool_calls", 2, "name"), "lookup") in events
