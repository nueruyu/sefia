"""The CLI-side input core.

Pending prompts, provided inputs, and queued inputs are persisted through a
:class:`KeyValueStore`, so a paused CLI invocation can be resumed by a later
one. The channel only sees primitives; how the runtime provides persistence
(and which tool raises the pause) is wired up by the integration layer.
"""

import inspect
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TypeVar

from ._kv import KeyValueStore
from .exceptions import AmbiguousInputError, UnknownInputError

_DEFAULT_NAMESPACE = "input_channel"

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]
InputRequestHandler = Callable[["InputRequest"], MaybeAwaitable[None]]
InputPromptDeltaHandler = Callable[[str], MaybeAwaitable[None]]


@dataclass(frozen=True)
class InputRequest:
    """A pending request for external input."""

    interaction_id: str
    prompt: str


class InputChannel:
    """The input pipe between a CLI application and a paused agent.

    One object owns the whole lifecycle. The tool-facing side records prompts
    and picks up provided input (:meth:`record_request` / :meth:`provide_input`
    / :meth:`complete_request`); the application-facing side routes arriving
    input to pending requests (:meth:`receive_input`); :meth:`use_store` binds
    the persistence both sides share. ``on_request`` and ``on_prompt_delta``
    are the rendering hooks (e.g. a :class:`CLIReporter`).

    Reads observe writes made earlier in the same session because the bound
    :class:`KeyValueStore` is expected to provide read-your-writes consistency.
    The active binding is held in a :class:`~contextvars.ContextVar` rather
    than a plain attribute so that a single shared channel stays correct when
    several sessions run concurrently: each task binds and reads its own store.
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
        self._on_request = on_request
        self._on_prompt_delta = on_prompt_delta

    @contextmanager
    def use_store(self, store: KeyValueStore):
        """Bind the persistence backing this channel for the enclosed block."""
        token = self._active_store.set(store)
        try:
            yield
        finally:
            self._active_store.reset(token)

    async def pending(self) -> list[InputRequest]:
        """The requests still waiting for input, ordered by interaction id."""
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
        """Route CLI input to a pending request, or queue it for the next one.

        ``None`` and blank input are ignored. With ``reply_to`` the input
        resolves that specific request; otherwise a single pending request is
        resolved directly, multiple pending requests raise
        :class:`AmbiguousInputError`, and no pending request queues the input
        for the next prompt.
        """
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

    async def provide_input(self, interaction_id: str) -> str | None:
        """Return the stored input, or claim a queued one if unambiguous."""
        provided = await self._stored_input(interaction_id)
        if provided is not None:
            return provided

        pending = await self._pending_map()
        if any(other_id != interaction_id for other_id in pending):
            return None

        return await self._pop_queued_input()

    async def record_request(self, interaction_id: str, prompt: str) -> None:
        pending = await self._pending_map()
        pending[interaction_id] = {"id": interaction_id, "prompt": prompt}
        await self._save_pending(pending)
        if self._on_request is not None:
            await _maybe_await(
                self._on_request(
                    InputRequest(interaction_id=interaction_id, prompt=prompt)
                )
            )

    async def complete_request(self, interaction_id: str) -> None:
        pending = await self._pending_map()
        pending.pop(interaction_id, None)
        await self._save_pending(pending)

    async def notify_prompt_delta(self, text: str) -> None:
        if self._on_prompt_delta is not None:
            await _maybe_await(self._on_prompt_delta(text))

    async def _pending_map(self) -> dict[str, dict]:
        store = self._store()
        pending = await store.get(self._pending_key, dict) or {}
        if not pending:
            return {}

        unresolved = {}
        for interaction_id, request in pending.items():
            provided = await self._stored_input(interaction_id)
            if provided is None:
                unresolved[interaction_id] = request

        await self._save_pending(unresolved)
        return dict(unresolved)

    async def _save_pending(self, pending: dict[str, dict]) -> None:
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
        queue = await store.get(self._queued_key, list) or []
        queue.append(input_text)
        await store.set(self._queued_key, queue, list)

    async def _pop_queued_input(self) -> str | None:
        store = self._store()
        queue = await store.get(self._queued_key, list)
        if not queue:
            return None

        next_input = queue.pop(0)
        if queue:
            await store.set(self._queued_key, queue, list)
        else:
            await store.delete(self._queued_key)
        return next_input

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


async def _maybe_await(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value
