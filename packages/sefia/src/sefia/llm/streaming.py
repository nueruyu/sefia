from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from typing_extensions import TypeAlias, final

from .json_schema import SchemaPath


@final
@dataclass(frozen=True)
class StringDelta:
    path: SchemaPath
    text: str


@final
@dataclass(frozen=True)
class StringEnd:
    path: SchemaPath
    value: str


@final
@dataclass(frozen=True)
class Scalar:
    path: SchemaPath
    value: int | float | bool | None


OutputEvent: TypeAlias = StringDelta | StringEnd | Scalar
OutputCallback: TypeAlias = Callable[[OutputEvent], Awaitable[None]]


__all__ = [
    "OutputCallback",
    "OutputEvent",
    "Scalar",
    "StringDelta",
    "StringEnd",
]
