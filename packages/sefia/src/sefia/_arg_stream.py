"""Routes streamed LLM response tokens into tool argument-stream handlers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from jsonstream import IncrementalJsonParser
from jsonstream import events as js

from .streaming import (
    ArgEvent,
    Scalar,
    StreamHandler,
    StringDelta,
    StringEnd,
    _ArgStreamChannel,
)

logger = logging.getLogger(__name__)

_HANDLER_DRAIN_TIMEOUT = 1.0


@dataclass(frozen=True)
class ToolCallPath:
    index: int
    field: Literal["name", "argument"]
    argument_name: str | None = None


def parse_tool_call_path(path: js.JsonPath) -> ToolCallPath | None:
    if len(path) < 3 or path[0] != "tool_calls":
        return None

    index = path[1]
    if not isinstance(index, int):
        raise TypeError(f"Expected integer tool call index, got {type(index).__name__}")

    field = path[2]
    if len(path) == 3 and field == "name":
        return ToolCallPath(index=index, field="name")

    if len(path) == 4 and field == "arguments":
        argument_name = path[3]
        if isinstance(argument_name, str):
            return ToolCallPath(
                index=index,
                field="argument",
                argument_name=argument_name,
            )

    return None


async def _run_handler(handler: StreamHandler, channel: _ArgStreamChannel) -> None:
    await handler(channel)


class ToolArgStreamer:
    """Feeds incrementally decoded tool arguments to per-tool stream handlers."""

    def __init__(self, tool_stream_handlers: Mapping[str, StreamHandler]) -> None:
        self._tool_stream_handlers = tool_stream_handlers
        self._reset()

    def _reset(self) -> None:
        self._parser = IncrementalJsonParser()
        self._stopped = False
        self._index_to_name: dict[int, str] = {}
        self._buffers: dict[int, list[ArgEvent]] = {}
        self._channels: dict[int, _ArgStreamChannel] = {}
        self._tasks: list[asyncio.Task[None]] = []

    def on_token(self, token: str) -> None:
        if self._stopped:
            return

        try:
            for event in self._parser.feed(token):
                self._dispatch(event)
                if self._stopped:
                    self._close_all_channels()
                    break
        except Exception:
            self._stopped = True
            self._close_all_channels()

    async def close(self) -> None:
        self._close_all_channels()
        tasks = self._tasks
        self._reset()

        if not tasks:
            return

        _, pending = await asyncio.wait(tasks, timeout=_HANDLER_DRAIN_TIMEOUT)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _dispatch(self, event: js.Event) -> None:
        if isinstance(event, js.JsonParseError):
            if event.fatal:
                self._stopped = True
            return

        path = getattr(event, "path", None)
        if path is None:
            return

        tool_path = parse_tool_call_path(path)
        if tool_path is None:
            return

        if tool_path.field == "name":
            if isinstance(event, js.EndString):
                self._resolve_tool_name(tool_path.index, event.value)
            return

        arg_event = _to_arg_event(tool_path, event)
        if arg_event is None:
            return

        channel = self._channels.get(tool_path.index)
        if channel is not None:
            channel.feed(arg_event)
        elif tool_path.index not in self._index_to_name:
            self._buffers.setdefault(tool_path.index, []).append(arg_event)

    def _resolve_tool_name(self, index: int, name: str) -> None:
        self._index_to_name[index] = name
        buffered = self._buffers.pop(index, [])

        handler = self._tool_stream_handlers.get(name)
        if handler is None:
            return

        channel = _ArgStreamChannel()
        self._channels[index] = channel
        task = asyncio.create_task(_run_handler(handler, channel))
        task.add_done_callback(self._log_task_result)
        self._tasks.append(task)
        for arg_event in buffered:
            channel.feed(arg_event)

    def _close_all_channels(self) -> None:
        for channel in self._channels.values():
            channel.close()
        self._channels.clear()
        self._buffers.clear()

    @staticmethod
    def _log_task_result(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Tool argument stream handler raised; ignoring.",
                exc_info=(type(error), error, error.__traceback__),
            )


def _to_arg_event(tool_path: ToolCallPath, event: js.Event) -> ArgEvent | None:
    name = tool_path.argument_name
    if name is None:
        return None

    if isinstance(event, js.StringDelta):
        return StringDelta(name=name, text=event.text)
    if isinstance(event, js.EndString):
        return StringEnd(name=name, value=event.value)
    if isinstance(event, js.Scalar):
        return Scalar(name=name, value=cast("int | float | bool | None", event.value))
    return None
