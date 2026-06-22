from jsonstream._strings import JsonStringDecoder, StringComplete
from jsonstream.events import JsonParseError, StringDelta


def feed_chars(decoder: JsonStringDecoder, text: str, *, emit_delta: bool = True):
    events = []
    for char in text:
        events.extend(decoder.feed(char, (0,), emit_delta=emit_delta))
    return events


def test_flush_delta_emits_and_preserves_final_value():
    decoder = JsonStringDecoder()

    assert feed_chars(decoder, "hello") == []
    assert list(decoder.flush_delta((0,), emit_delta=True)) == [
        StringDelta(path=(0,), text="hello")
    ]
    assert feed_chars(decoder, '!"') == [
        StringDelta(path=(0,), text="!"),
        StringComplete("hello!"),
    ]


def test_finish_can_suppress_key_deltas():
    decoder = JsonStringDecoder()

    assert feed_chars(decoder, 'key"', emit_delta=False) == [StringComplete("key")]


def test_escape_splits_delta_before_decoded_character():
    decoder = JsonStringDecoder()

    assert feed_chars(decoder, "a\\n", emit_delta=True) == [
        StringDelta(path=(0,), text="a")
    ]
    assert feed_chars(decoder, '"', emit_delta=True) == [
        StringDelta(path=(0,), text="\n"),
        StringComplete("a\n"),
    ]


def test_chunked_unicode_escape_waits_until_complete():
    decoder = JsonStringDecoder()

    assert feed_chars(decoder, "\\u12") == []
    assert not decoder.can_flush_delta
    assert feed_chars(decoder, '34"') == [
        StringDelta(path=(0,), text="\u1234"),
        StringComplete("\u1234"),
    ]


def test_surrogate_pair_decodes_single_character():
    decoder = JsonStringDecoder()

    assert feed_chars(decoder, '\\uD83D\\uDC4B"') == [
        StringDelta(path=(0,), text="\U0001f44b"),
        StringComplete("\U0001f44b"),
    ]


def test_unescaped_control_character_is_fatal():
    decoder = JsonStringDecoder()

    events = feed_chars(decoder, "\n")

    assert events == [JsonParseError("Invalid control character in string", fatal=True)]
