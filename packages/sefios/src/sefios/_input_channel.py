"""Persisted routing between sefios input tools and host integrations."""

import asyncio
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, TypeAlias, TypedDict, TypeVar, cast

from ._async import MaybeAwaitable, maybe_await
from ._input import InputRequest
from .exceptions import AmbiguousInputError, UnknownInputError

_DEFAULT_NAMESPACE = "input_channel"

T = TypeVar("T")
InputRequestHandler = Callable[["InputRequest"], MaybeAwaitable[None]]
InputPromptDeltaHandler = Callable[[str, str], MaybeAwaitable[None]]


class _PendingRequest(TypedDict):
    id: str
    prompt: str


_PendingMap: TypeAlias = dict[str, _PendingRequest]


class KeyValueStore(Protocol):
    """Async persistence required by an input channel."""

    async def get(self, key: str, type_hint: type[T]) -> T | None: ...

    async def set(self, key: str, value: Any, type_hint: type[Any]) -> None: ...

    async def delete(self, key: str) -> None: ...


class InputChannel:
    """Routes external input to persisted pending requests.

    A channel binds persistence per context so one shared instance remains safe
    across concurrent sessions. Sibling tasks in one binding serialize channel
    updates; separate bindings must not concurrently write the same session.
    Optional callbacks let an adapter observe new requests and streamed prompt
    text without changing the routing state machine.
    """

    def __init__(
        self,
        *,
        on_request: InputRequestHandler | None = None,
        on_prompt_delta: InputPromptDeltaHandler | None = None,
        namespace: str = _DEFAULT_NAMESPACE,
    ):
        namespace = namespace.strip("/")
        if not namespace:
            raise ValueError("Input channel namespace must not be empty.")
        self._namespace = namespace
        self._active_store: ContextVar[KeyValueStore | None] = ContextVar(
            "input_active_store", default=None
        )
        self._active_lock: ContextVar[asyncio.Lock] = ContextVar("input_lock")
        self._on_request = on_request
        self._on_prompt_delta = on_prompt_delta

    @contextmanager
    def use_store(self, store: KeyValueStore) -> Generator[None]:
        """Bind the persistence backing this channel for the enclosed block."""
        token = self._active_store.set(store)
        lock_token = self._active_lock.set(asyncio.Lock())
        channel_token = _active_channel.set(self)
        try:
            yield
        finally:
            _active_channel.reset(channel_token)
            self._active_lock.reset(lock_token)
            self._active_store.reset(token)

    async def pending(self) -> list[InputRequest]:
        """Return requests still waiting for input, ordered by interaction id."""
        async with self._lock():
            pending = await self._pending_map()
            return [
                InputRequest(interaction_id=entry["id"], prompt=entry["prompt"])
                for _, entry in sorted(pending.items())
            ]

    async def receive_input(
        self,
        input_value: str | list[str] | None,
        *,
        reply_to: str | None = None,
    ) -> None:
        """Route input to a pending request, or queue it for the next one."""
        async with self._lock():
            if input_value is None:
                return
            input_text = _to_input_text(input_value)
            if not input_text:
                return

            pending = await self._pending_map()

            if reply_to is not None:
                if reply_to not in pending:
                    raise UnknownInputError(reply_to)
                await self._store_input(reply_to, input_text)
                return

            if len(pending) == 1:
                await self._store_input(next(iter(pending)), input_text)
                return

            if len(pending) > 1:
                raise AmbiguousInputError(sorted(pending))

            await self._queue_input(input_text)

    async def provide_input(
        self, interaction_id: str, *, allow_queued: bool = True
    ) -> str | None:
        """Return stored input, or claim a queued input when unambiguous."""
        async with self._lock():
            provided = await self._stored_input(interaction_id)
            if provided is not None:
                return provided

            if not allow_queued:
                return None

            pending = await self._pending_map()
            if any(other_id != interaction_id for other_id in pending):
                return None

            return await self._pop_queued_input()

    async def record_request(self, interaction_id: str, prompt: str) -> None:
        async with self._lock():
            pending = await self._pending_map()
            pending[interaction_id] = {"id": interaction_id, "prompt": prompt}
            await self._save_pending(pending)
        if self._on_request is not None:
            await maybe_await(
                self._on_request(
                    InputRequest(interaction_id=interaction_id, prompt=prompt)
                )
            )

    async def complete_request(self, interaction_id: str) -> None:
        async with self._lock():
            pending = await self._pending_map()
            pending.pop(interaction_id, None)
            await self._save_pending(pending)

    async def notify_prompt_delta(self, interaction_id: str, text: str) -> None:
        if self._on_prompt_delta is not None:
            await maybe_await(self._on_prompt_delta(interaction_id, text))

    async def _pending_map(self) -> _PendingMap:
        store = self._store()
        pending = cast(_PendingMap | None, await store.get(self._pending_key, dict))
        if not pending:
            return {}

        unresolved: _PendingMap = {}
        for interaction_id, request in pending.items():
            provided = await self._stored_input(interaction_id)
            if provided is None:
                unresolved[interaction_id] = request

        await self._save_pending(unresolved)
        return dict(unresolved)

    async def _save_pending(self, pending: _PendingMap) -> None:
        store = self._store()
        if pending:
            await store.set(self._pending_key, pending, dict)
            return
        await store.delete(self._pending_key)

    async def _stored_input(self, interaction_id: str) -> str | None:
        return await self._store().get(self._input_key(interaction_id), str)

    async def _store_input(self, interaction_id: str, input_text: str) -> None:
        await self._store().set(self._input_key(interaction_id), input_text, str)

    async def _queue_input(self, input_text: str) -> None:
        store = self._store()
        queue = cast(list[str] | None, await store.get(self._queued_key, list)) or []
        queue.append(input_text)
        await store.set(self._queued_key, queue, list)

    async def _pop_queued_input(self) -> str | None:
        store = self._store()
        queue = cast(list[str] | None, await store.get(self._queued_key, list))
        if not queue:
            return None

        next_input = queue.pop(0)
        if queue:
            await store.set(self._queued_key, queue, list)
        else:
            await store.delete(self._queued_key)
        return next_input

    def _lock(self) -> asyncio.Lock:
        self._store()
        return self._active_lock.get()

    def _store(self) -> KeyValueStore:
        store = self._active_store.get()
        if store is None:
            raise RuntimeError("Input channel is not bound to a store.")
        return store

    @property
    def _pending_key(self) -> str:
        return f"{self._namespace}/pending"

    @property
    def _queued_key(self) -> str:
        return f"{self._namespace}/queued"

    def _input_key(self, interaction_id: str) -> str:
        return f"{self._namespace}/input/{interaction_id}"


def _to_input_text(input_value: str | list[str]) -> str:
    if isinstance(input_value, str):
        return input_value.strip()
    return " ".join(input_value).strip()


_active_channel: ContextVar[InputChannel] = ContextVar("active_input_channel")


def get_input_channel() -> InputChannel:
    try:
        return _active_channel.get()
    except LookupError:
        raise RuntimeError(
            "External input requires an active sefios session."
        ) from None
