from jsonweir import IncrementalJsonParser
from jsonweir import events as js

from sefia.llm.streaming import (
    StructuredOutputCallback,
    StructuredOutputEvent,
    StructuredScalar,
    StructuredStringDelta,
    StructuredStringEnd,
)

from ._adapter import LiteLLMPreparedSchema


class StructuredOutputStreamer:
    def __init__(
        self,
        prepared: LiteLLMPreparedSchema,
        callback: StructuredOutputCallback,
    ) -> None:
        self._prepared = prepared
        self._callback = callback
        self._parser = IncrementalJsonParser()

    async def feed(self, token: str) -> None:
        for event in self._parser.feed(token):
            converted = self._convert(event)
            if converted is not None:
                await self._callback(converted)

    def _convert(self, event: js.Event) -> StructuredOutputEvent | None:
        path = getattr(event, "path", None)
        if path is None:
            return None
        logical_path = self._prepared.normalize_stream_path(path)
        if logical_path is None:
            return None
        if isinstance(event, js.StringDelta):
            return StructuredStringDelta(logical_path, event.text)
        if isinstance(event, js.EndString):
            return StructuredStringEnd(logical_path, event.value)
        if isinstance(event, js.Scalar):
            if isinstance(event.value, str):
                return None
            return StructuredScalar(logical_path, event.value)
        return None
