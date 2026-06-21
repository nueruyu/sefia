from jsonstream import (
    EndArray,
    EndObject,
    EndString,
    IncrementalJsonParser,
    JsonParseError,
    Key,
    Scalar,
    StartArray,
    StartObject,
    StartString,
    StringDelta,
)


def run_parser(chunks: list[str]):
    parser = IncrementalJsonParser()
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    return events


def test_empty_object():
    events = run_parser(["{}"])
    assert events == [
        StartObject(path=()),
        EndObject(path=()),
    ]


def test_empty_array():
    events = run_parser(["[]"])
    assert events == [
        StartArray(path=()),
        EndArray(path=()),
    ]


def test_simple_object():
    events = run_parser(['{"key": "value"}'])
    assert events == [
        StartObject(path=()),
        Key(path=(), value="key"),
        StartString(path=("key",)),
        StringDelta(path=("key",), text="value"),
        EndString(path=("key",), value="value"),
        EndObject(path=("key",)),
    ]


def test_simple_array():
    events = run_parser(["[true, null, 123]"])
    assert events == [
        StartArray(path=()),
        Scalar(path=(0,), value=True),
        Scalar(path=(1,), value=None),
        Scalar(path=(2,), value=123),
        EndArray(path=(3,)),
    ]


def test_nested_structure():
    events = run_parser(['{"a": [1, {"b": "c"}]}'])
    assert events == [
        StartObject(path=()),
        Key(path=(), value="a"),
        StartArray(path=("a",)),
        Scalar(path=("a", 0), value=1),
        StartObject(path=("a", 1)),
        Key(path=("a", 1), value="b"),
        StartString(path=("a", 1, "b")),
        StringDelta(path=("a", 1, "b"), text="c"),
        EndString(path=("a", 1, "b"), value="c"),
        EndObject(path=("a", 1, "b")),
        EndArray(path=("a", 2)),
        EndObject(path=("a",)),
    ]


def test_chunked_string_value():
    events = run_parser(['{"key": "hello', ' world"}'])
    assert events == [
        StartObject(path=()),
        Key(path=(), value="key"),
        StartString(path=("key",)),
        StringDelta(path=("key",), text="hello"),
        StringDelta(path=("key",), text=" world"),
        EndString(path=("key",), value="hello world"),
        EndObject(path=("key",)),
    ]


def test_chunked_literal():
    events = run_parser(["[tr", "ue, fa", "lse, nu", "ll]"])
    assert events == [
        StartArray(path=()),
        Scalar(path=(0,), value=True),
        Scalar(path=(1,), value=False),
        Scalar(path=(2,), value=None),
        EndArray(path=(3,)),
    ]


def test_chunked_number():
    events = run_parser(["[12", "3, 4", "5.67]"])
    assert events == [
        StartArray(path=()),
        Scalar(path=(0,), value=123),
        Scalar(path=(1,), value=45.67),
        EndArray(path=(2,)),
    ]


def test_empty_string():
    events = run_parser(['[""]'])
    assert events == [
        StartArray(path=()),
        StartString(path=(0,)),
        EndString(path=(0,), value=""),
        EndArray(path=(1,)),
    ]


def test_string_with_standard_escapes():
    events = run_parser(['["\\"\\\\\\/\\b\\f\\n\\r\\t"]'])
    expected_text = '"\\/\b\f\n\r\t'
    assert events == [
        StartArray(path=()),
        StartString(path=(0,)),
        StringDelta(path=(0,), text='"'),
        StringDelta(path=(0,), text="\\"),
        StringDelta(path=(0,), text="/"),
        StringDelta(path=(0,), text="\b"),
        StringDelta(path=(0,), text="\f"),
        StringDelta(path=(0,), text="\n"),
        StringDelta(path=(0,), text="\r"),
        StringDelta(path=(0,), text="\t"),
        EndString(path=(0,), value=expected_text),
        EndArray(path=(1,)),
    ]


def test_chunked_escape_sequence():
    events = run_parser(['["\\', 'n"]'])
    assert events == [
        StartArray(path=()),
        StartString(path=(0,)),
        StringDelta(path=(0,), text="\n"),
        EndString(path=(0,), value="\n"),
        EndArray(path=(1,)),
    ]


def test_unicode_bmp():
    events = run_parser(['["\\u1234"]'])
    assert events == [
        StartArray(path=()),
        StartString(path=(0,)),
        StringDelta(path=(0,), text="\u1234"),
        EndString(path=(0,), value="\u1234"),
        EndArray(path=(1,)),
    ]


def test_unicode_surrogate_pair():
    events = run_parser(['["\\uD83D\\uDC4B"]'])
    expected = "\U0001f44b"
    assert events == [
        StartArray(path=()),
        StartString(path=(0,)),
        StringDelta(path=(0,), text=expected),
        EndString(path=(0,), value=expected),
        EndArray(path=(1,)),
    ]


def test_chunked_surrogate_pair():
    events = run_parser(['["\\uD83D', '\\uDC4B"]'])
    expected = "\U0001f44b"
    assert events == [
        StartArray(path=()),
        StartString(path=(0,)),
        StringDelta(path=(0,), text=expected),
        EndString(path=(0,), value=expected),
        EndArray(path=(1,)),
    ]


def test_fatal_unterminated_string():
    events = run_parser(['{"key": "value'])
    assert any(isinstance(event, JsonParseError) and event.fatal for event in events)


def test_fatal_unterminated_object():
    events = run_parser(['{"key": "value"'])
    assert any(isinstance(event, JsonParseError) and event.fatal for event in events)


def test_fatal_unexpected_character():
    events = run_parser(["[1, 2, ?]"])
    assert events[:2] == [StartArray(path=()), Scalar(path=(0,), value=1)]
    assert any(
        isinstance(event, JsonParseError)
        and event.fatal
        and "Invalid literal" in event.message
        for event in events
    )


def test_fatal_invalid_escape():
    events = run_parser(['["\\q"]'])
    assert any(
        isinstance(event, JsonParseError)
        and event.fatal
        and "Invalid escape" in event.message
        for event in events
    )


def test_fatal_missing_colon():
    events = run_parser(['{"key" "value"}'])
    assert any(
        isinstance(event, JsonParseError)
        and event.fatal
        and "Unexpected string" in event.message
        for event in events
    )


def test_fatal_high_surrogate_not_followed_by_low():
    events = run_parser(['["\\uD83D\\u1234"]'])
    assert any(
        isinstance(event, JsonParseError)
        and event.fatal
        and "not followed by low surrogate" in event.message
        for event in events
    )


def test_fatal_high_surrogate_not_followed_by_escape():
    events = run_parser(['["\\uD83D\\n"]'])
    assert any(
        isinstance(event, JsonParseError)
        and event.fatal
        and "not followed by low surrogate" in event.message
        for event in events
    )
