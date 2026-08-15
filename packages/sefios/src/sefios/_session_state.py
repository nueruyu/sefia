"""The session-scoped state binding and its accessor functions.

This is the low-level, string-keyed tier of the state API, meant for tool
implementations: :func:`get_call_state_store` for call-scoped resumable state,
:func:`get_session_storage` for raw key-value access. Application and handler
state should normally go through the type-keyed container returned by
:func:`sefios.get_state` instead.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from collections.abc import Generator
from contextlib import contextmanager
from typing import Type, TypeVar

from glyff import ExecutionId  # pyright: ignore[reportMissingTypeStubs]
from glyff import (  # pyright: ignore[reportMissingTypeStubs]
    get_context as get_glyff_context,
)
from glyff.exceptions import ContextNotSetError  # pyright: ignore[reportMissingTypeStubs]

from ._state_store import StateStore
from .storage import SessionStorage

T = TypeVar("T")


def _execution_id_to_data(execution_id: ExecutionId) -> dict[str, object]:
    parent_id = execution_id.parent_id
    return {
        "domain_id": execution_id.domain_id.value,
        "name": execution_id.name.value,
        "sequence": execution_id.sequence,
        "arguments_digest": execution_id.arguments_digest.value,
        "parent_id": _execution_id_to_data(parent_id) if parent_id else None,
    }


def _execution_id_scope_key(execution_id: ExecutionId) -> str:
    data = _execution_id_to_data(execution_id)
    stable_repr = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_repr.encode("utf-8")).hexdigest()


class _SessionState:
    """Internal per-session binding over a :class:`SessionStorage`.

    Hands out typed, caching :class:`StateStore` views of the storage — either
    keyed directly (session scope) or scoped to the current engraved call (call
    scope). Not exported: callers use the module-level accessor functions.
    """

    def __init__(self, storage: SessionStorage):
        self._storage = storage
        self._state_stores: dict[str, StateStore] = {}

    @property
    def storage(self) -> SessionStorage:
        """The underlying :class:`SessionStorage`."""
        return self._storage

    def get_call_state_store(
        self, key_suffix: str, state_type: Type[T]
    ) -> StateStore[T]:
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
        if key not in self._state_stores:
            self._state_stores[key] = StateStore(
                storage=self._storage,
                key=key,
                state_type=state_type,
            )
        store = self._state_stores[key]
        if store.state_type != state_type:
            raise TypeError(
                f"State store for key '{key}' was already created with a different type."
            )
        return store


_session_state_var = contextvars.ContextVar[_SessionState]("sefios_session_state")


def _get_session_state() -> _SessionState:
    try:
        return _session_state_var.get()
    except LookupError:
        raise RuntimeError(
            "No sefios session state is bound. Are you running outside a "
            "SessionScope.session() (or bind_session_storage) block?"
        ) from None


def get_session_storage() -> SessionStorage:
    """Returns the :class:`SessionStorage` bound to the current session.

    This is the raw key-value escape hatch for application code that manages
    its own keys. Raises ``RuntimeError`` if called outside an active session
    (for example, outside ``SessionScope.session()``).
    """
    return _get_session_state().storage


def get_call_state_store(key_suffix: str, state_type: Type[T]) -> StateStore[T]:
    """Returns a :class:`StateStore` scoped to the current engraved call.

    The store's key is derived from the call's execution id, so a resumed
    invocation that re-enters the same engraved call reads back the same state
    it stored before pausing. Raises ``RuntimeError`` outside an engraved
    function or an active session.
    """
    return _get_session_state().get_call_state_store(key_suffix, state_type)


def get_state_store(key: str, state_type: Type[T]) -> StateStore[T]:
    """Returns the session-scoped :class:`StateStore` for ``key``.

    Prefer the type-keyed :func:`sefios.get_state` container in application
    code; this exists for callers that must manage string keys themselves.
    """
    return _get_session_state().get_state_store(key, state_type)


@contextmanager
def bind_session_storage(storage: SessionStorage) -> Generator[None]:
    """Binds ``storage`` as the current session's state storage."""
    token = _session_state_var.set(_SessionState(storage))
    try:
        yield
    finally:
        _session_state_var.reset(token)
