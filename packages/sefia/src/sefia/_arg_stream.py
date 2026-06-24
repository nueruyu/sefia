"""Routes a streaming LLM response into tool argument-stream handlers.

The model emits its decision as a single JSON object whose ``tool_calls`` carry
each call's ``name`` and ``arguments``. As tokens arrive (published as
``LLMTokenReceived``), this router parses them incrementally with
``jsonstream`` and forwards the values found under
``tool_calls[i].arguments.<name>`` to the stream handler registered for that
tool, as :mod:`sefia.streaming` events.

It is installed by the executor only when a tool actually registers a stream
handler, and runs as an ordinary (isolated) event handler — so any failure here
can never affect the inference outcome, which is decoded authoritatively from
the full response once it has arrived.
"""

from __future__ import annotations

import asyncio
from typing import cast

from jsonstream import IncrementalJsonParser
from jsonstream import events as js

from .event_system import EventHandler
from .llm.events import AfterLLMCall, BeforeLLMCall, LLMTokenReceived
from .streaming import (
    ArgEvent,
    Scalar,
    StreamHandler,
    StringDelta,
    StringEnd,
    _ArgStreamChannel,
)

_RouterEvent = LLMTokenReceived | AfterLLMCall | BeforeLLMCall


async def _run_handler(handler: StreamHandler, channel: _ArgStreamChannel) -> None:
    """Adapt a handler's ``Awaitable`` return to the coroutine create_task wants."""
    await handler(channel)


class ArgStreamRouter(EventHandler[_RouterEvent]):
    """Feeds incrementally decoded tool arguments to per-tool stream handlers."""

    def __init__(self, handlers_by_tool: dict[str, StreamHandler]) -> None:
        self._handlers_by_tool = handlers_by_tool
        self._reset()

    def _reset(self) -> None:
        self._parser = IncrementalJsonParser()
        self._stopped = False
        self._index_to_name: dict[int, str] = {}
        self._buffers: dict[int, list[ArgEvent]] = {}
        self._channels: dict[int, _ArgStreamChannel] = {}
        self._tasks: list[asyncio.Task[None]] = []

    async def handle(self, event: _RouterEvent) -> None:
        if isinstance(event, BeforeLLMCall):
            self._reset()
        elif isinstance(event, LLMTokenReceived):
            self._on_token(event.token)
        elif isinstance(event, AfterLLMCall):
            await self._finalize()

    def _on_token(self, token: str) -> None:
        if self._stopped:
            return
        # The side channel is best-effort: if the stream is not the bare JSON we
        # expect (e.g. wrapped in a markdown fence), stop quietly and let the
        # authoritative decode of the full response handle it.
        try:
            for event in self._parser.feed(token):
                self._dispatch(event)
                if self._stopped:
                    break
        except Exception:
            self._stopped = True

    def _dispatch(self, event: js.Event) -> None:
        if isinstance(event, js.JsonParseError):
            if event.fatal:
                self._stopped = True
            return

        if isinstance(event, js.EndString) and _is_tool_name_path(event.path):
            self._resolve_tool_name(_index_of(event.path), event.value)
            return

        arg_event = _to_arg_event(event)
        if arg_event is None:
            return

        index = _index_of(event.path)
        channel = self._channels.get(index)
        if channel is not None:
            channel.feed(arg_event)
        elif index not in self._index_to_name:
            # The tool name has not been decoded yet, so we cannot tell which
            # handler (if any) this belongs to. Buffer until the name resolves.
            self._buffers.setdefault(index, []).append(arg_event)
        # else: name is known but has no handler — drop.

    def _resolve_tool_name(self, index: int, name: str) -> None:
        self._index_to_name[index] = name
        buffered = self._buffers.pop(index, [])

        handler = self._handlers_by_tool.get(name)
        if handler is None:
            return

        channel = _ArgStreamChannel()
        self._channels[index] = channel
        self._tasks.append(asyncio.create_task(_run_handler(handler, channel)))
        for arg_event in buffered:
            channel.feed(arg_event)

    async def _finalize(self) -> None:
        for channel in self._channels.values():
            channel.close()
        tasks = self._tasks
        self._reset()
        if tasks:
            # Wait for handlers to drain so live output completes before the
            # tool itself runs. Handler failures are isolated, not propagated.
            await asyncio.gather(*tasks, return_exceptions=True)


def _is_tool_name_path(path: js.JsonPath) -> bool:
    return len(path) == 3 and path[0] == "tool_calls" and path[2] == "name"


def _index_of(path: js.JsonPath) -> int:
    index = path[1]
    assert isinstance(index, int)
    return index


def _to_arg_event(event: js.Event) -> ArgEvent | None:
    path = getattr(event, "path", None)
    if path is None:
        return None
    if not (len(path) == 4 and path[0] == "tool_calls" and path[2] == "arguments"):
        return None
    name = path[3]
    if not isinstance(name, str):
        return None

    if isinstance(event, js.StringDelta):
        return StringDelta(name=name, text=event.text)
    if isinstance(event, js.EndString):
        return StringEnd(name=name, value=event.value)
    if isinstance(event, js.Scalar):
        # jsonstream emits Scalar only for non-string scalars; strings arrive as
        # StringDelta/EndString. Narrow away the str that JsonScalar permits.
        return Scalar(name=name, value=cast("int | float | bool | None", event.value))
    return None
