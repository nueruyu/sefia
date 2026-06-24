from collections.abc import Generator

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
    Event,
    JsonParseError,
    JsonPath,
    StartArray,
    StartObject,
)


class ContainerTracker:
    def __init__(self) -> None:
        self._stack: list[ContainerState] = [ArrayState()]

    @property
    def path(self) -> JsonPath:
        return path_from_stack(self._stack[1:])

    @property
    def has_unclosed_containers(self) -> bool:
        return len(self._stack) > 1

    @property
    def root_is_empty(self) -> bool:
        root = self._stack[0]
        return isinstance(root, ArrayState) and root.next_index == 0

    def is_expecting_value(self) -> bool:
        return self._stack[-1].state in (
            ParserState.EXPECT_VALUE,
            ParserState.EXPECT_VALUE_OR_ARRAY_END,
        )

    def is_expecting_object_key(self) -> bool:
        current_container = self._stack[-1]
        return (
            isinstance(current_container, ObjectState)
            and current_container.state == ParserState.EXPECT_KEY_OR_OBJECT_END
        )

    def set_object_key(self, key: str) -> bool:
        current_container = self._stack[-1]
        if not isinstance(current_container, ObjectState):
            return False

        current_container.current_key = key
        current_container.state = ParserState.EXPECT_COLON
        return True

    def value_completed(self) -> None:
        parent_container = self._stack[-1]
        if isinstance(parent_container, ArrayState):
            parent_container.next_index += 1
            parent_container.state = ParserState.EXPECT_COMMA_OR_ARRAY_END
        else:
            parent_container.member_count += 1
            parent_container.state = ParserState.EXPECT_COMMA_OR_OBJECT_END

    def parse_structural_char(self, char: str) -> Generator[Event, None, None]:
        current_container = self._stack[-1]
        state = current_container.state

        if char == "{":
            if not self.is_expecting_value():
                yield JsonParseError("Unexpected '{'", fatal=True)
                return
            yield StartObject(path=self.path)
            self._stack.append(ObjectState())
        elif char == "}":
            yield from self._end_object(current_container, state)
        elif char == "[":
            if not self.is_expecting_value():
                yield JsonParseError("Unexpected '['", fatal=True)
                return
            yield StartArray(path=self.path)
            self._stack.append(ArrayState())
        elif char == "]":
            yield from self._end_array(current_container, state)
        elif char == ":":
            if (
                not isinstance(current_container, ObjectState)
                or state != ParserState.EXPECT_COLON
            ):
                yield JsonParseError("Unexpected ':'", fatal=True)
                return
            current_container.state = ParserState.EXPECT_VALUE
        elif char == ",":
            yield from self._parse_comma(current_container, state)

    def _end_object(
        self, current_container: ContainerState, state: ParserState
    ) -> Generator[Event, None, None]:
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

        path_before_pop = path_from_stack(self._stack[1:-1])
        self._stack.pop()
        yield EndObject(path=path_before_pop)
        self.value_completed()

    def _end_array(
        self, current_container: ContainerState, state: ParserState
    ) -> Generator[Event, None, None]:
        if len(self._stack) == 1:
            yield JsonParseError("Unexpected ']'", fatal=True)
            return
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

        path_before_pop = path_from_stack(self._stack[1:-1])
        self._stack.pop()
        yield EndArray(path=path_before_pop)
        self.value_completed()

    def _parse_comma(
        self, current_container: ContainerState, state: ParserState
    ) -> Generator[Event, None, None]:
        if len(self._stack) == 1:
            # The root is wrapped in a virtual array; a comma here would
            # otherwise allow multiple top-level values or a trailing comma,
            # both of which are invalid per RFC 8259.
            yield JsonParseError("Unexpected ','", fatal=True)
        elif (
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
