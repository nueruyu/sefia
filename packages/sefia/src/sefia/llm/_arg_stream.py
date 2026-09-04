"""Routes logical structured-output events into tool argument stream handlers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from sefia.streaming import ArgStream

from ..streaming import ArgEvent, StreamHandler

logger = logging.getLogger(__name__)

_HANDLER_DRAIN_TIMEOUT = 1.0


_CLOSED = object()


class _ArgStreamChannel:
    """An async iterator the router feeds while the stream handler consumes it.

    Backed by an unbounded queue so the (synchronous) parser can push events
    without awaiting. ``close`` ends iteration once every queued event has been
    delivered.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[ArgEvent | object] = asyncio.Queue()
        self._closed = False
        self._done = False

    def __aiter__(self) -> ArgStream:
        return self

    async def __anext__(self) -> ArgEvent:
        if self._done:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is _CLOSED:
            self._done = True
            raise StopAsyncIteration
        return item  # type: ignore[return-value]

    def feed(self, event: ArgEvent) -> None:
        if not self._closed:
            self._queue.put_nowait(event)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(_CLOSED)


async def _run_handler(
    handler: StreamHandler, tool_call_id: str, channel: _ArgStreamChannel
) -> None:
    await handler(tool_call_id, channel)


class ToolArgStreamer:
    """Feeds incrementally decoded tool arguments to per-tool stream handlers."""

    def __init__(
        self,
        tool_stream_handlers: Mapping[str, StreamHandler],
        get_tool_call_id: Callable[[int], str],
    ) -> None:
        self._tool_stream_handlers = tool_stream_handlers
        self._get_tool_call_id = get_tool_call_id
        self._reset()

    def _reset(self) -> None:
        self._index_to_name: dict[int, str] = {}
        self._buffers: dict[int, list[ArgEvent]] = {}
        self._channels: dict[int, _ArgStreamChannel] = {}
        self._tasks: list[asyncio.Task[None]] = []

    def identify_tool(self, index: int, name: str) -> None:
        self._resolve_tool_name(index, name)

    def on_argument(self, index: int, event: ArgEvent) -> None:
        channel = self._channels.get(index)
        if channel is not None:
            channel.feed(event)
        elif index not in self._index_to_name:
            self._buffers.setdefault(index, []).append(event)

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

    def _resolve_tool_name(self, index: int, name: str) -> None:
        if index in self._index_to_name:
            return
        self._index_to_name[index] = name
        buffered = self._buffers.pop(index, [])

        handler = self._tool_stream_handlers.get(name)
        if handler is None:
            return

        channel = _ArgStreamChannel()
        self._channels[index] = channel
        tool_call_id = self._get_tool_call_id(index)
        task = asyncio.create_task(_run_handler(handler, tool_call_id, channel))
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
