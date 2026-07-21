"""Opinionated stack for building applications with the Sefia framework."""

from glyff import engrave
from sefia import AsRawText, Policy, Profile, Tools, infer, policy, preview, profile

from ._scope import SessionScope
from ._session_state import get_call_state_store, get_session_storage
from ._state_store import StateStore
from .exceptions import NeedsInput
from .state import StateContainer, StateRegistry, get_state, state
from .storage import FileSessionStorage, MemorySessionStorage, SessionStorage

__all__ = [
    # Authoring surface re-exported from the core, so app code only imports `sefios`.
    "infer",
    "preview",
    "policy",
    "profile",
    "Policy",
    "Profile",
    "Tools",
    "AsRawText",
    "engrave",
    # sefios' own front door and batteries.
    "SessionScope",
    "NeedsInput",
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
