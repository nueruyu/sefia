from jsonweir import IncrementalJsonParser
from jsonweir import events as js

from sefia.llm.streaming import (
    OutputStreamCallback,
    OutputStreamEvent,
    Scalar,
    StringDelta,
    StringEnd,
)

from ._schema import DecisionEnvelopeFormat


class OutputEventStreamer:
    def __init__(
        self,
        schema: DecisionEnvelopeFormat,
        callback: OutputStreamCallback,
    ) -> None:
        self._schema = schema
        self._callback = callback
        self._parser = IncrementalJsonParser()

    async def feed(self, token: str) -> None:
        for event in self._parser.feed(token):
            converted = self._convert(event)
            if converted is not None:
                await self._callback(converted)

    def _convert(self, event: js.Event) -> OutputStreamEvent | None:
        path = getattr(event, "path", None)
        if path is None:
            return None
        payload_path = self._schema.to_payload_path(path)
        if payload_path is None:
            return None
        if isinstance(event, js.StringDelta):
            return StringDelta(payload_path, event.text)
        if isinstance(event, js.EndString):
            return StringEnd(payload_path, event.value)
        if isinstance(event, js.Scalar):
            if isinstance(event.value, str):
                return None
            return Scalar(payload_path, event.value)
        return None
