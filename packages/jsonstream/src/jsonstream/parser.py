from collections.abc import Generator

from ._literals import decode_literal
from ._state import (
    ArrayState,
    ContainerState,
    ObjectState,
    ParserState,
    path_from_stack,
)
from .events import (
    EndArray,
    EndObject,
    EndString,
    Event,
    JsonParseError,
    JsonPath,
    Key,
    Scalar,
    StartArray,
    StartObject,
    StartString,
    StringDelta,
)


class IncrementalJsonParser:
    def __init__(self) -> None:
        self._container_stack: list[ContainerState] = [ArrayState()]

        self._literal_buffer: list[str] = []
        self._string_buffer: list[str] = []
        self._full_string_value: list[str] = []

        self._in_string = False
        self._string_is_key = False
        self._in_escape = False
        self._unicode_buffer: list[str] | None = None
        self._high_surrogate: int | None = None

    @property
    def path(self) -> JsonPath:
        return path_from_stack(self._container_stack[1:])

    def feed(self, chunk: str) -> Generator[Event, None, None]:
        for char in chunk:
            yield from self._parse_char(char)

        if self._in_string and not self._in_escape and self._unicode_buffer is None:
            yield from self._flush_string_delta_if_any()

    def finish(self) -> Generator[Event, None, None]:
        yield from self._flush_literal_buffer_if_any()

        if self._in_string:
            yield JsonParseError("Unterminated string at end of input", fatal=True)
            return

        if len(self._container_stack) > 1:
            yield JsonParseError(
                "Unterminated object or array at end of input", fatal=True
            )
            return

        root = self._container_stack[0]
        if isinstance(root, ArrayState) and root.next_index == 0:
            yield JsonParseError("Expected JSON value at end of input", fatal=True)

    def _parse_char(self, char: str) -> Generator[Event, None, None]:
        if self._in_string:
            yield from self._parse_string_char(char)
            return

        if char.isspace():
            yield from self._flush_literal_buffer_if_any()
            return

        if char == '"':
            yield from self._flush_literal_buffer_if_any()
            yield from self._start_string()
        elif char in "{}[]:,":
            yield from self._flush_literal_buffer_if_any()
            yield from self._parse_structural_char(char)
        else:
            self._literal_buffer.append(char)

    def _flush_literal_buffer_if_any(self) -> Generator[Event, None, None]:
        if not self._literal_buffer:
            return

        literal = "".join(self._literal_buffer)
        self._literal_buffer.clear()

        current_state = self._container_stack[-1].state
        if current_state not in (
            ParserState.EXPECT_VALUE,
            ParserState.EXPECT_VALUE_OR_ARRAY_END,
        ):
            yield JsonParseError(f"Unexpected literal '{literal}'", fatal=True)
            return

        try:
            value = decode_literal(literal)
        except ValueError:
            yield JsonParseError(f"Invalid literal or number: {literal}", fatal=True)
            return

        yield Scalar(path=self.path, value=value)
        self._transition_state_after_value()

    def _parse_structural_char(self, char: str) -> Generator[Event, None, None]:
        current_container = self._container_stack[-1]
        state = current_container.state

        if char == "{":
            if state not in (
                ParserState.EXPECT_VALUE,
                ParserState.EXPECT_VALUE_OR_ARRAY_END,
            ):
                yield JsonParseError("Unexpected '{'", fatal=True)
                return
            yield StartObject(path=self.path)
            self._container_stack.append(ObjectState())
        elif char == "}":
            if not isinstance(current_container, ObjectState):
                yield JsonParseError("Unexpected '}'", fatal=True)
                return
            if state == ParserState.EXPECT_KEY_OR_OBJECT_END:
                if current_container.member_count != 0:
                    yield JsonParseError("Unexpected '}'", fatal=True)
                    return
            elif state != ParserState.EXPECT_COMMA_OR_OBJECT_END:
                yield JsonParseError("Unexpected '}'", fatal=True)
                return

            path_before_pop = self.path
            self._container_stack.pop()
            yield EndObject(path=path_before_pop)
            self._transition_state_after_value()
        elif char == "[":
            if state not in (
                ParserState.EXPECT_VALUE,
                ParserState.EXPECT_VALUE_OR_ARRAY_END,
            ):
                yield JsonParseError("Unexpected '['", fatal=True)
                return
            yield StartArray(path=self.path)
            self._container_stack.append(ArrayState())
        elif char == "]":
            if not isinstance(current_container, ArrayState):
                yield JsonParseError("Unexpected ']'", fatal=True)
                return
            if state == ParserState.EXPECT_VALUE_OR_ARRAY_END:
                if current_container.next_index != 0:
                    yield JsonParseError("Unexpected ']'", fatal=True)
                    return
            elif state != ParserState.EXPECT_COMMA_OR_ARRAY_END:
                yield JsonParseError("Unexpected ']'", fatal=True)
                return

            path_before_pop = self._end_array_path(current_container)
            self._container_stack.pop()
            yield EndArray(path=path_before_pop)
            self._transition_state_after_value()
        elif char == ":":
            if (
                not isinstance(current_container, ObjectState)
                or state != ParserState.EXPECT_COLON
            ):
                yield JsonParseError("Unexpected ':'", fatal=True)
                return
            current_container.state = ParserState.EXPECT_VALUE
        elif char == ",":
            if (
                isinstance(current_container, ObjectState)
                and state == ParserState.EXPECT_COMMA_OR_OBJECT_END
            ):
                current_container.current_key = None
                current_container.state = ParserState.EXPECT_KEY_OR_OBJECT_END
            elif (
                isinstance(current_container, ArrayState)
                and state == ParserState.EXPECT_COMMA_OR_ARRAY_END
            ):
                current_container.state = ParserState.EXPECT_VALUE
            else:
                yield JsonParseError("Unexpected ','", fatal=True)

    def _end_array_path(self, current_container: ArrayState) -> JsonPath:
        if current_container.next_index == 0:
            return path_from_stack(self._container_stack[1:-1])
        return self.path

    def _start_string(self) -> Generator[Event, None, None]:
        current_container = self._container_stack[-1]
        state = current_container.state

        if (
            isinstance(current_container, ObjectState)
            and state == ParserState.EXPECT_KEY_OR_OBJECT_END
        ):
            self._in_string = True
            self._string_is_key = True
        elif state in (ParserState.EXPECT_VALUE, ParserState.EXPECT_VALUE_OR_ARRAY_END):
            self._in_string = True
            self._string_is_key = False
            yield StartString(path=self.path)
        else:
            yield JsonParseError("Unexpected string", fatal=True)

    def _end_string(self) -> Generator[Event, None, None]:
        if self._high_surrogate is not None:
            yield JsonParseError(
                "High surrogate not followed by low surrogate.", fatal=True
            )
            self._high_surrogate = None

        yield from self._flush_string_delta_if_any()

        self._in_string = False
        self._in_escape = False
        final_value = "".join(self._full_string_value)
        self._full_string_value.clear()

        current_container = self._container_stack[-1]
        if self._string_is_key:
            if not isinstance(current_container, ObjectState):
                yield JsonParseError("Unexpected object key", fatal=True)
                return
            yield Key(path=self.path, value=final_value)
            current_container.current_key = final_value
            current_container.state = ParserState.EXPECT_COLON
        else:
            yield EndString(path=self.path, value=final_value)
            self._transition_state_after_value()

        self._string_is_key = False

    def _parse_string_char(self, char: str) -> Generator[Event, None, None]:
        if self._in_escape:
            yield from self._handle_escape_char(char)
        elif char == "\\":
            yield from self._flush_string_delta_if_any()
            self._in_escape = True
        elif char == '"':
            yield from self._end_string()
        else:
            if self._high_surrogate is not None:
                yield JsonParseError(
                    "High surrogate not followed by low surrogate.", fatal=True
                )
                self._high_surrogate = None
            self._string_buffer.append(char)

    def _handle_escape_char(self, char: str) -> Generator[Event, None, None]:
        if self._unicode_buffer is not None:
            self._unicode_buffer.append(char)
            if len(self._unicode_buffer) == 4:
                yield from self._decode_unicode()
            return

        decoded: str | None = None
        if char == '"':
            decoded = '"'
        elif char == "\\":
            decoded = "\\"
        elif char == "/":
            decoded = "/"
        elif char == "b":
            decoded = "\b"
        elif char == "f":
            decoded = "\f"
        elif char == "n":
            decoded = "\n"
        elif char == "r":
            decoded = "\r"
        elif char == "t":
            decoded = "\t"
        elif char == "u":
            self._unicode_buffer = []
            return
        else:
            yield JsonParseError(f"Invalid escape sequence '\\{char}'", fatal=True)
            decoded = char

        yield from self._append_decoded_string_char(decoded)
        self._in_escape = False

    def _decode_unicode(self) -> Generator[Event, None, None]:
        if self._unicode_buffer is None:
            return

        hex_digits = "".join(self._unicode_buffer)
        self._unicode_buffer = None
        self._in_escape = False

        try:
            codepoint = int(hex_digits, 16)
        except ValueError:
            yield JsonParseError(
                f"Invalid unicode escape sequence '\\u{hex_digits}'", fatal=True
            )
            return

        if 0xD800 <= codepoint <= 0xDBFF:
            if self._high_surrogate is not None:
                yield JsonParseError("Unexpected high surrogate pair.", fatal=True)
            self._high_surrogate = codepoint
        elif 0xDC00 <= codepoint <= 0xDFFF:
            if self._high_surrogate is None:
                yield JsonParseError(
                    "Unexpected low surrogate without a high surrogate.", fatal=True
                )
                return

            combined = 0x10000 + (
                ((self._high_surrogate - 0xD800) << 10) | (codepoint - 0xDC00)
            )
            self._string_buffer.append(chr(combined))
            self._high_surrogate = None
        else:
            if self._high_surrogate is not None:
                yield JsonParseError(
                    "High surrogate not followed by low surrogate.", fatal=True
                )
                self._high_surrogate = None
            self._string_buffer.append(chr(codepoint))

    def _append_decoded_string_char(self, char: str) -> Generator[Event, None, None]:
        if self._high_surrogate is not None:
            yield JsonParseError(
                "High surrogate not followed by low surrogate.", fatal=True
            )
            self._high_surrogate = None
        self._string_buffer.append(char)

    def _flush_string_delta_if_any(self) -> Generator[Event, None, None]:
        if not self._string_buffer:
            return

        delta = "".join(self._string_buffer)
        self._string_buffer.clear()
        self._full_string_value.append(delta)

        if not self._string_is_key:
            yield StringDelta(path=self.path, text=delta)

    def _transition_state_after_value(self) -> None:
        parent_container = self._container_stack[-1]
        if isinstance(parent_container, ArrayState):
            parent_container.next_index += 1
            parent_container.state = ParserState.EXPECT_COMMA_OR_ARRAY_END
        else:
            parent_container.member_count += 1
            parent_container.state = ParserState.EXPECT_COMMA_OR_OBJECT_END
