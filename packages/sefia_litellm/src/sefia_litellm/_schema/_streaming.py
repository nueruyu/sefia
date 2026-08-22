from jsonweir import IncrementalJsonParser
from jsonweir import events as js

from sefia.llm.streaming import (
    OutputCallback,
    OutputEvent,
    Scalar,
    StringDelta,
    StringEnd,
)

from ._codec import PreparedOutput


class OutputEventStreamer:
    def __init__(
        self,
        prepared: PreparedOutput,
        callback: OutputCallback,
    ) -> None:
        self._prepared = prepared
        self._callback = callback
        self._parser = IncrementalJsonParser()

    async def feed(self, token: str) -> None:
        for event in self._parser.feed(token):
            converted = self._convert(event)
            if converted is not None:
                await self._callback(converted)

    def _convert(self, event: js.Event) -> OutputEvent | None:
        path = getattr(event, "path", None)
        if path is None:
            return None
        logical_path = self._prepared.logical_path(path)
        if logical_path is None:
            return None
        if isinstance(event, js.StringDelta):
            return StringDelta(logical_path, event.text)
        if isinstance(event, js.EndString):
            return StringEnd(logical_path, event.value)
        if isinstance(event, js.Scalar):
            if isinstance(event.value, str):
                return None
            return Scalar(logical_path, event.value)
        return None
