"""The HTTP-side human-in-the-loop core.

Pending prompts, provided inputs, and queued inputs are persisted through a
:class:`KeyValueStore`, so a paused request can be resumed by a later one. The
store only sees primitives; how the runtime provides persistence (and which
tool raises the pause) is wired up by the integration layer.

Deliberately independent from the CLI counterpart in ``sefia_typer``: the two
surfaces share semantics today but are free to diverge.
"""

from contextlib import contextmanager
from contextvars import ContextVar

from ._kv import KeyValueStore
from .exceptions import AmbiguousInputError, UnknownInputError

_PENDING_INPUTS_KEY = "pending_inputs"
_QUEUED_INPUTS_KEY = "queued_inputs"


class InputStore:
    """Small persistence wrapper for input state in the active session.

    Reads observe writes made earlier in the same session because the bound
    :class:`KeyValueStore` is expected to provide read-your-writes consistency.

    The active binding is held in a :class:`~contextvars.ContextVar` rather than a
    plain attribute so that a single shared store instance stays correct when
    several sessions run concurrently (e.g. one asyncio task per HTTP request):
    each task binds and reads its own store.
    """

    def __init__(self):
        self._active_store: ContextVar[KeyValueStore | None] = ContextVar(
            "input_active_store", default=None
        )

    @contextmanager
    def use_store(self, store: KeyValueStore):
        token = self._active_store.set(store)
        try:
            yield
        finally:
            self._active_store.reset(token)

    async def pending_requests(self) -> dict[str, dict]:
        store = self._store()
        pending = await store.get(_PENDING_INPUTS_KEY, dict) or {}
        if not pending:
            return {}

        unresolved = {}
        for interaction_id, request in pending.items():
            provided = await self.get_input(interaction_id)
            if provided is None:
                unresolved[interaction_id] = request

        await self.save_pending_requests(unresolved)
        return dict(unresolved)

    async def save_pending_requests(self, pending: dict[str, dict]) -> None:
        store = self._store()
        if pending:
            await store.set(_PENDING_INPUTS_KEY, pending, dict)
            return

        await store.delete(_PENDING_INPUTS_KEY)

    async def get_input(self, interaction_id: str) -> str | None:
        return await self._store().get(self._input_key(interaction_id), str)

    async def set_input(self, interaction_id: str, input_text: str) -> None:
        await self._store().set(self._input_key(interaction_id), input_text, str)

    async def queue_input(self, input_text: str) -> None:
        store = self._store()
        queue = await store.get(_QUEUED_INPUTS_KEY, list) or []
        queue.append(input_text)
        await store.set(_QUEUED_INPUTS_KEY, queue, list)

    async def pop_queued_input(self) -> str | None:
        store = self._store()
        queue = await store.get(_QUEUED_INPUTS_KEY, list)
        if not queue:
            return None

        next_input = queue.pop(0)
        if queue:
            await store.set(_QUEUED_INPUTS_KEY, queue, list)
        else:
            await store.delete(_QUEUED_INPUTS_KEY)
        return next_input

    def _store(self) -> KeyValueStore:
        store = self._active_store.get()
        if store is None:
            raise RuntimeError("Input store is not bound to a session.")
        return store

    @staticmethod
    def _input_key(interaction_id: str) -> str:
        return f"input__{interaction_id}"


class InputReceiver:
    """Accepts request input and stores it for pending or future inputs."""

    def __init__(self, store: InputStore):
        self._store = store

    async def receive_input(
        self,
        input_value: str | list[str] | None,
        *,
        reply_to: str | None = None,
    ) -> None:
        """Route request input to a pending prompt, or queue it for the next.

        ``None`` and blank input are ignored. With ``reply_to`` the input
        resolves that specific request; otherwise a single pending request is
        resolved directly, multiple pending requests raise
        :class:`AmbiguousInputError`, and no pending request queues the
        input for the next prompt.
        """
        if input_value is None:
            return
        input_text = _to_input_text(input_value)
        if not input_text:
            return

        pending = await self._store.pending_requests()

        if reply_to is not None:
            if reply_to not in pending:
                raise UnknownInputError(reply_to)
            await self._store.set_input(reply_to, input_text)
            return

        if len(pending) == 1:
            await self._store.set_input(next(iter(pending)), input_text)
            return

        if len(pending) > 1:
            raise AmbiguousInputError(sorted(pending))

        await self._store.queue_input(input_text)


class InputCoordinator:
    """Serves a input tool's lifecycle from the HTTP input store.

    The methods take and return primitives, so the integration layer can wire
    them to whichever tool implementation raises the pause.
    """

    def __init__(self, store: InputStore | None = None):
        self.store = store or InputStore()

    async def provide_input(self, interaction_id: str) -> str | None:
        """Return the stored input, or claim a queued one if unambiguous."""
        provided = await self.store.get_input(interaction_id)
        if provided is not None:
            return provided

        pending = await self.store.pending_requests()
        if any(other_id != interaction_id for other_id in pending):
            return None

        return await self.store.pop_queued_input()

    async def record_request(self, interaction_id: str, prompt: str) -> None:
        pending = await self.store.pending_requests()
        pending[interaction_id] = {"id": interaction_id, "prompt": prompt}
        await self.store.save_pending_requests(pending)

    async def complete_request(self, interaction_id: str) -> None:
        pending = await self.store.pending_requests()
        pending.pop(interaction_id, None)
        await self.store.save_pending_requests(pending)


def _to_input_text(input_value: str | list[str]) -> str:
    if isinstance(input_value, str):
        return input_value.strip()
    return " ".join(input_value).strip()
