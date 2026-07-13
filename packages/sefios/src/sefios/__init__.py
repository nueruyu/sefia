"""Official stack for building applications with the Sefia framework."""

from ._scope import SessionScope
from ._session_state import get_call_state_store, get_session_storage
from ._state_store import StateStore
from .exceptions import NeedsInput
from .history_storages import SessionHistoryStorage
from .state import StateContainer, StateRegistry, get_state, state
from .storage import FileSessionStorage, MemorySessionStorage, SessionStorage

__all__ = [
    "SessionScope",
    "NeedsInput",
    "SessionHistoryStorage",
    "SessionStorage",
    "StateStore",
    "MemorySessionStorage",
    "FileSessionStorage",
    "get_call_state_store",
    "get_session_storage",
    "StateContainer",
    "StateRegistry",
    "get_state",
    "state",
]
