"""Opinionated stack for building applications with the Sefia framework."""

from sefia import (
    AsRawText,
    Policy,
    Profile,
    Tools,
    concurrent,
    policy,
    preview,
    profile,
)

from ._scope import SessionScope
from ._domain import domain
from ._session_state import get_call_state_store, get_session_storage
from ._state_store import StateStore
from .persistence import (
    FilePersistence,
    MemoryPersistence,
    SessionPersistence,
    SQLitePersistence,
)
from .state import StateContainer, StateRegistry, get_state, state
from .storage import (
    FileSessionStorage,
    MemorySessionStorage,
    SessionStorage,
    SQLiteSessionStorage,
)

__all__ = [
    # Authoring surface re-exported from the core, so app code only imports `sefios`.
    "concurrent",
    "preview",
    "policy",
    "profile",
    "Policy",
    "Profile",
    "Tools",
    "AsRawText",
    "domain",
    # sefios' own front door and batteries.
    "SessionScope",
    "SessionPersistence",
    "SQLitePersistence",
    "MemoryPersistence",
    "FilePersistence",
    "SessionStorage",
    "StateStore",
    "MemorySessionStorage",
    "FileSessionStorage",
    "SQLiteSessionStorage",
    "get_call_state_store",
    "get_session_storage",
    "StateContainer",
    "StateRegistry",
    "get_state",
    "state",
]
