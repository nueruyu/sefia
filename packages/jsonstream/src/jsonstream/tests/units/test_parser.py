import pytest

from jsonstream import IncrementalJsonParser
from jsonstream.events import (
    EndArray,
    EndObject,
    EndString,
    JsonParseError,
    Key,
    Scalar,
    StartArray,
    StartObject,
    StartString,
    StringDelta,
)


def fatal_messages(events):
    return [
        event.message
        for event in events
        if isinstance(event, JsonParseError) and event.fatal
    ]


def assert_fatal_error(events, message_part: str) -> None:
    assert any(message_part in message for message in fatal_messages(events))


def run_parser(chunks: list[str]):
    parser = IncrementalJsonParser()
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    return events


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        (["123"], [Scalar(path=(), value=123)]),
        (
            ['"abc"'],
            [
                StartString(path=()),
                StringDelta(path=(), text="abc"),
                EndString(path=(), value="abc"),
            ],
        ),
    ],
)
def test_root_scalar_values(chunks, expected):
    assert run_parser(chunks) == expected


def test_whitespace_only_is_fatal():
    events = run_parser(["   \n\t"])
    assert_fatal_error(events, "Expected JSON value")


@pytest.mark.parametrize("whitespace", ["\v", "\f", "\u00a0"])
def test_non_rfc_whitespace_is_not_treated_as_whitespace(whitespace):
    # Only space, tab, LF, and CR are JSON whitespace; other Unicode
    # whitespace must be rejected rather than silently skipped.
    events = run_parser([f"{whitespace}1"])
    assert_fatal_error(events, "Invalid literal or number")


def test_public_api_exports_parser_only():
    import jsonstream

    assert jsonstream.IncrementalJsonParser is IncrementalJsonParser
    assert not hasattr(jsonstream, "StartObject")
    assert not hasattr(jsonstream, "JsonParseError")


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
        EndObject(path=()),
    ]


def test_simple_array():
    events = run_parser(["[true, null, 123]"])
    assert events == [
        StartArray(path=()),
        Scalar(path=(0,), value=True),
        Scalar(path=(1,), value=None),
        Scalar(path=(2,), value=123),
        EndArray(path=()),
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
        EndObject(path=("a", 1)),
        EndArray(path=("a",)),
        EndObject(path=()),
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
        EndObject(path=()),
    ]


def test_chunked_object_key():
    events = run_parser(['{"na', 'me": 1}'])
    assert events == [
        StartObject(path=()),
        Key(path=(), value="name"),
        Scalar(path=("name",), value=1),
        EndObject(path=()),
    ]


def test_chunked_literal():
    events = run_parser(["[tr", "ue, fa", "lse, nu", "ll]"])
    assert events == [
        StartArray(path=()),
        Scalar(path=(0,), value=True),
        Scalar(path=(1,), value=False),
        Scalar(path=(2,), value=None),
        EndArray(path=()),
    ]


def test_chunked_number():
    events = run_parser(["[12", "3, 4", "5.67]"])
    assert events == [
        StartArray(path=()),
        Scalar(path=(0,), value=123),
        Scalar(path=(1,), value=45.67),
        EndArray(path=()),
    ]


def test_empty_string():
    events = run_parser(['[""]'])
    assert events == [
        StartArray(path=()),
        StartString(path=(0,)),
        EndString(path=(0,), value=""),
        EndArray(path=()),
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
        EndArray(path=()),
    ]


def test_chunked_escape_sequence():
    events = run_parser(['["\\', 'n"]'])
    assert events == [
        StartArray(path=()),
        StartString(path=(0,)),
        StringDelta(path=(0,), text="\n"),
        EndString(path=(0,), value="\n"),
        EndArray(path=()),
    ]


def test_unicode_bmp():
    events = run_parser(['["\\u1234"]'])
    assert events == [
        StartArray(path=()),
        StartString(path=(0,)),
        StringDelta(path=(0,), text="\u1234"),
        EndString(path=(0,), value="\u1234"),
        EndArray(path=()),
    ]


def test_unicode_surrogate_pair():
    events = run_parser(['["\\uD83D\\uDC4B"]'])
    expected = "\U0001f44b"
    assert events == [
        StartArray(path=()),
        StartString(path=(0,)),
        StringDelta(path=(0,), text=expected),
        EndString(path=(0,), value=expected),
        EndArray(path=()),
    ]


def test_chunked_surrogate_pair():
    events = run_parser(['["\\uD83D', '\\uDC4B"]'])
    expected = "\U0001f44b"
    assert events == [
        StartArray(path=()),
        StartString(path=(0,)),
        StringDelta(path=(0,), text=expected),
        EndString(path=(0,), value=expected),
        EndArray(path=()),
    ]


def test_chunked_unicode_escape_digits():
    events = run_parser(['["\\u', '1234"]'])
    assert events == [
        StartArray(path=()),
        StartString(path=(0,)),
        StringDelta(path=(0,), text="\u1234"),
        EndString(path=(0,), value="\u1234"),
        EndArray(path=()),
    ]


@pytest.mark.parametrize(
    "source",
    [
        '["\\u+123"]',
        '["\\u 123"]',
        '["\\u-100"]',
        '["\\u12gh"]',
        '["\\u12"]',
        '["\\u1"]',
    ],
)
def test_fatal_invalid_unicode_escape_digits(source):
    events = run_parser([source])
    assert_fatal_error(events, "unicode escape sequence")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("[-1]", -1),
        ("[0]", 0),
        ("[1.2]", 1.2),
        ("[-1.2e+3]", -1200.0),
    ],
)
def test_valid_number_forms(source, expected):
    events = run_parser([source])
    assert events == [
        StartArray(path=()),
        Scalar(path=(0,), value=expected),
        EndArray(path=()),
    ]


@pytest.mark.parametrize("source", ["[01]", "[1.]", "[1e]", "[--1]"])
def test_invalid_number_forms_are_fatal(source):
    events = run_parser([source])
    assert_fatal_error(events, "Invalid literal or number")


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


@pytest.mark.parametrize(
    ("source", "message_part"),
    [
        ("[1,]", "Unexpected ']'"),
        ('{"a":1,}', "Unexpected '}'"),
        ("[1 2]", "Unexpected literal"),
        ('{"a":1 "b":2}', "Unexpected string"),
    ],
)
def test_fatal_separator_errors(source, message_part):
    events = run_parser([source])
    assert_fatal_error(events, message_part)


@pytest.mark.parametrize("source", ["{}[]", "{}1"])
def test_fatal_multiple_top_level_values(source):
    events = run_parser([source])
    assert fatal_messages(events)


@pytest.mark.parametrize("source", ["1,2", "1,", "{},{}", "[],[]", '"a","b"'])
def test_fatal_root_comma_separated_values(source):
    events = run_parser([source])
    assert_fatal_error(events, "Unexpected ','")


def test_fatal_unescaped_control_character_in_string():
    events = run_parser(['["a\n"]'])
    assert_fatal_error(events, "Invalid control character")
