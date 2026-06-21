import dataclasses
from enum import Enum, auto
from typing import TypeAlias

from .events import JsonPath


class ParserState(Enum):
    EXPECT_VALUE_OR_ARRAY_END = auto()
    EXPECT_KEY_OR_OBJECT_END = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    EXPECT_COMMA_OR_ARRAY_END = auto()
    EXPECT_COMMA_OR_OBJECT_END = auto()


@dataclasses.dataclass
class ArrayState:
    state: ParserState = ParserState.EXPECT_VALUE_OR_ARRAY_END
    next_index: int = 0


@dataclasses.dataclass
class ObjectState:
    state: ParserState = ParserState.EXPECT_KEY_OR_OBJECT_END
    current_key: str | None = None
    member_count: int = 0


ContainerState: TypeAlias = ArrayState | ObjectState


def path_from_stack(stack: list[ContainerState]) -> JsonPath:
    path: list[str | int] = []
    for container in stack:
        if isinstance(container, ObjectState):
            if container.current_key is not None:
                path.append(container.current_key)
        else:
            path.append(container.next_index)
    return tuple(path)
