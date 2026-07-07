"""Official stack for building applications with the Sefia framework."""

from ._scope import SessionScope
from ._session_state import SessionState, get_session_state
from ._state_store import StateStore
from .exceptions import NeedsInput
from .state import StateContainer, StateRegistry, get_state, state
from .stores import FileSessionStore, MemorySessionStore, SessionStore

__all__ = [
    "SessionScope",
    "NeedsInput",
    "SessionState",
    "SessionStore",
    "StateStore",
    "MemorySessionStore",
    "FileSessionStore",
    "get_session_state",
    "StateContainer",
    "StateRegistry",
    "get_state",
    "state",
]
