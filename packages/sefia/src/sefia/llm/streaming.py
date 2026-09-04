from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from jsonweir import IncrementalJsonParser
from jsonweir import events as js
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


OutputStreamEvent: TypeAlias = StringDelta | StringEnd | Scalar
OutputStreamCallback: TypeAlias = Callable[[OutputStreamEvent], Awaitable[None]]


@final
class JsonOutputStreamDecoder:
    """Decodes streamed JSON text into logical output events."""

    def __init__(self) -> None:
        self._parser = IncrementalJsonParser()

    def feed(self, token: str) -> list[OutputStreamEvent]:
        return [
            converted
            for event in self._parser.feed(token)
            if (converted := _convert_event(event)) is not None
        ]


def _convert_event(event: js.Event) -> OutputStreamEvent | None:
    path = getattr(event, "path", None)
    if path is None:
        return None
    if isinstance(event, js.StringDelta):
        return StringDelta(path, event.text)
    if isinstance(event, js.EndString):
        return StringEnd(path, event.value)
    if isinstance(event, js.Scalar):
        if isinstance(event.value, str):
            return None
        return Scalar(path, event.value)
    return None


__all__ = [
    "JsonOutputStreamDecoder",
    "OutputStreamCallback",
    "OutputStreamEvent",
    "Scalar",
    "StringDelta",
    "StringEnd",
]
