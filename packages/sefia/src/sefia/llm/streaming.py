from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from typing_extensions import TypeAlias, final

from .schema import SchemaPath


@final
@dataclass(frozen=True)
class StructuredStringDelta:
    path: SchemaPath
    text: str


@final
@dataclass(frozen=True)
class StructuredStringEnd:
    path: SchemaPath
    value: str


@final
@dataclass(frozen=True)
class StructuredScalar:
    path: SchemaPath
    value: int | float | bool | None


StructuredOutputEvent: TypeAlias = (
    StructuredStringDelta | StructuredStringEnd | StructuredScalar
)
StructuredOutputCallback: TypeAlias = Callable[[StructuredOutputEvent], Awaitable[None]]


__all__ = [
    "StructuredOutputCallback",
    "StructuredOutputEvent",
    "StructuredScalar",
    "StructuredStringDelta",
    "StructuredStringEnd",
]
