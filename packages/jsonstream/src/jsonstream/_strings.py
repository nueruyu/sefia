import dataclasses
from collections.abc import Generator
from typing import TypeAlias

from .events import JsonParseError, JsonPath, StringDelta


@dataclasses.dataclass(frozen=True)
class StringComplete:
    value: str


StringDecoderEvent: TypeAlias = StringDelta | JsonParseError | StringComplete


class JsonStringDecoder:
    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._value_parts: list[str] = []
        self._in_escape = False
        self._unicode_buffer: list[str] | None = None
        self._high_surrogate: int | None = None

    @property
    def can_flush_delta(self) -> bool:
        return not self._in_escape and self._unicode_buffer is None

    def feed(
        self, char: str, path: JsonPath, *, emit_delta: bool
    ) -> Generator[StringDecoderEvent, None, None]:
        if self._in_escape:
            yield from self._handle_escape_char(char)
        elif char == "\\":
            yield from self.flush_delta(path, emit_delta=emit_delta)
            self._in_escape = True
        elif char == '"':
            yield from self.finish(path, emit_delta=emit_delta)
        else:
            if ord(char) < 0x20:
                yield JsonParseError("Invalid control character in string", fatal=True)
                return
            if self._high_surrogate is not None:
                yield JsonParseError(
                    "High surrogate not followed by low surrogate.", fatal=True
                )
                self._high_surrogate = None
            self._buffer.append(char)

    def finish(
        self, path: JsonPath, *, emit_delta: bool
    ) -> Generator[StringDecoderEvent, None, None]:
        if self._high_surrogate is not None:
            yield JsonParseError(
                "High surrogate not followed by low surrogate.", fatal=True
            )
            self._high_surrogate = None

        yield from self.flush_delta(path, emit_delta=emit_delta)

        final_value = "".join(self._value_parts)
        self._value_parts.clear()
        yield StringComplete(final_value)

    def flush_delta(
        self, path: JsonPath, *, emit_delta: bool
    ) -> Generator[StringDelta, None, None]:
        if not self._buffer:
            return

        delta = "".join(self._buffer)
        self._buffer.clear()
        self._value_parts.append(delta)

        if emit_delta:
            yield StringDelta(path=path, text=delta)

    def _handle_escape_char(self, char: str) -> Generator[JsonParseError, None, None]:
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

        yield from self._append_decoded_char(decoded)
        self._in_escape = False

    def _decode_unicode(self) -> Generator[JsonParseError, None, None]:
        if self._unicode_buffer is None:
            return

        hex_digits = "".join(self._unicode_buffer)
        self._unicode_buffer = None
        self._in_escape = False

        # int(..., 16) also accepts signs and surrounding whitespace (e.g.
        # "+12" or " 12"), but RFC 8259 requires exactly four hex digits, so
        # validate the characters explicitly before decoding.
        if not all(c in "0123456789abcdefABCDEF" for c in hex_digits):
            yield JsonParseError(
                f"Invalid unicode escape sequence '\\u{hex_digits}'", fatal=True
            )
            return

        codepoint = int(hex_digits, 16)

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
            self._buffer.append(chr(combined))
            self._high_surrogate = None
        else:
            if self._high_surrogate is not None:
                yield JsonParseError(
                    "High surrogate not followed by low surrogate.", fatal=True
                )
                self._high_surrogate = None
            self._buffer.append(chr(codepoint))

    def _append_decoded_char(self, char: str) -> Generator[JsonParseError, None, None]:
        if self._high_surrogate is not None:
            yield JsonParseError(
                "High surrogate not followed by low surrogate.", fatal=True
            )
            self._high_surrogate = None
        self._buffer.append(char)
