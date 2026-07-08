from __future__ import annotations

import contextvars
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Type, TypeVar

from glyff import ExecutionId
from glyff import get_context as get_glyff_context
from glyff.exceptions import ContextNotSetError

from ._state_store import StateStore
from .stores import SessionStore

T = TypeVar("T")


def _execution_id_to_data(execution_id: ExecutionId) -> dict[str, object]:
    parent_id = execution_id.parent_id
    return {
        "name": execution_id.name,
        "sequence": execution_id.sequence,
        "args_hash": execution_id.args_hash,
        "parent_id": _execution_id_to_data(parent_id) if parent_id else None,
    }


def _execution_id_scope_key(execution_id: ExecutionId) -> str:
    data = _execution_id_to_data(execution_id)
    stable_repr = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_repr.encode("utf-8")).hexdigest()


class SessionState:
    """Session-scoped state persistence bound to a single active session.

    Holds the session's :class:`SessionStore` and hands out typed, caching
    :class:`StateStore` views of it — either keyed directly (session scope) or
    scoped to the current engraved call (call scope, for resumable tool state).

    This is the low-level, string-keyed tier of the state API, meant for tool
    implementations. Application and handler state should normally go through
    the type-keyed container returned by :func:`sefios.get_state` instead.
    """

    def __init__(self, store: SessionStore):
        self._store = store
        self._state_stores: dict[str, StateStore] = {}

    @property
    def store(self) -> SessionStore:
        """The underlying :class:`SessionStore`."""
        return self._store

    def get_call_state_store(
        self, key_suffix: str, state_type: Type[T]
    ) -> StateStore[T]:
        """
        Gets a StateStore scoped to the current engraved function call.
        This provides call-local state for a single invocation of an engraved tool.
        """
        try:
            glyff_ctx = get_glyff_context()
        except ContextNotSetError:
            glyff_ctx = None
        current_execution_id = (
            glyff_ctx.current_execution_id if glyff_ctx is not None else None
        )

        if current_execution_id is None:
            raise RuntimeError(
                "get_call_state_store can only be used inside an engraved function."
            )

        scope_key = _execution_id_scope_key(current_execution_id)
        scoped_key = f"call_state/{scope_key}/{key_suffix}"
        return self.get_state_store(scoped_key, state_type)

    def get_state_store(self, key: str, state_type: Type[T]) -> StateStore[T]:
        """Gets a StateStore for the given key and type, creating one if needed."""
        if key not in self._state_stores:
            self._state_stores[key] = StateStore(
                store=self._store,
                key=key,
                state_type=state_type,
            )
        store = self._state_stores[key]
        if store.state_type != state_type:
            raise TypeError(
                f"State store for key '{key}' was already created with a different type."
            )
        return store


_session_state_var = contextvars.ContextVar[SessionState]("sefios_session_state")


def get_session_state() -> SessionState:
    """Returns the :class:`SessionState` bound to the current session.

    Raises ``RuntimeError`` if called outside an active session (for example,
    outside ``SessionScope.session()`` or ``bind_session_state``).
    """
    try:
        return _session_state_var.get()
    except LookupError:
        raise RuntimeError(
            "No sefios session state is bound. Are you running outside a "
            "SessionScope.session() (or bind_session_state) block?"
        ) from None


@contextmanager
def bind_session_state(store: SessionStore) -> Iterator[SessionState]:
    """Binds a fresh :class:`SessionState` over ``store`` for the current context."""
    state = SessionState(store)
    token = _session_state_var.set(state)
    try:
        yield state
    finally:
        _session_state_var.reset(token)
