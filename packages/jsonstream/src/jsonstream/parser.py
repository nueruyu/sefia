from collections.abc import Generator

from ._containers import ContainerTracker
from ._literals import decode_literal
from ._strings import JsonStringDecoder, StringComplete
from .events import (
    EndString,
    Event,
    JsonParseError,
    JsonPath,
    Key,
    Scalar,
    StartString,
)


class IncrementalJsonParser:
    def __init__(self) -> None:
        self._containers = ContainerTracker()
        self._literal_buffer: list[str] = []
        self._string_decoder = JsonStringDecoder()

        self._in_string = False
        self._string_is_key = False

    @property
    def path(self) -> JsonPath:
        return self._containers.path

    def feed(self, chunk: str) -> Generator[Event, None, None]:
        for char in chunk:
            yield from self._parse_char(char)

        if self._in_string and self._string_decoder.can_flush_delta:
            yield from self._string_decoder.flush_delta(
                self.path, emit_delta=not self._string_is_key
            )

    def finish(self) -> Generator[Event, None, None]:
        yield from self._flush_literal_buffer_if_any()

        if self._in_string:
            yield JsonParseError("Unterminated string at end of input", fatal=True)
            return

        if self._containers.has_unclosed_containers:
            yield JsonParseError(
                "Unterminated object or array at end of input", fatal=True
            )
            return

        if self._containers.root_is_empty:
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

        if not self._containers.is_expecting_value():
            yield JsonParseError(f"Unexpected literal '{literal}'", fatal=True)
            return

        try:
            value = decode_literal(literal)
        except ValueError:
            yield JsonParseError(f"Invalid literal or number: {literal}", fatal=True)
            return

        yield Scalar(path=self.path, value=value)
        self._containers.value_completed()

    def _parse_structural_char(self, char: str) -> Generator[Event, None, None]:
        yield from self._containers.parse_structural_char(char)

    def _start_string(self) -> Generator[Event, None, None]:
        if self._containers.is_expecting_object_key():
            self._in_string = True
            self._string_is_key = True
            self._string_decoder = JsonStringDecoder()
        elif self._containers.is_expecting_value():
            self._in_string = True
            self._string_is_key = False
            self._string_decoder = JsonStringDecoder()
            yield StartString(path=self.path)
        else:
            yield JsonParseError("Unexpected string", fatal=True)

    def _end_string(self, final_value: str) -> Generator[Event, None, None]:
        self._in_string = False

        if self._string_is_key:
            key_path = self.path
            if not self._containers.set_object_key(final_value):
                yield JsonParseError("Unexpected object key", fatal=True)
                return
            yield Key(path=key_path, value=final_value)
        else:
            yield EndString(path=self.path, value=final_value)
            self._containers.value_completed()

        self._string_is_key = False

    def _parse_string_char(self, char: str) -> Generator[Event, None, None]:
        for event in self._string_decoder.feed(
            char, self.path, emit_delta=not self._string_is_key
        ):
            if isinstance(event, StringComplete):
                yield from self._end_string(event.value)
            else:
                yield event
