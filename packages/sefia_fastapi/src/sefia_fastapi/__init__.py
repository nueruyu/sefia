"""FastAPI (HTTP) building blocks for Sefia applications.

This package holds the HTTP-side pieces that depend only on ``sefia`` and
FastAPI: the human-in-the-loop core (persisted over a :class:`KeyValueStore`),
the session event broker with its SSE response helper, and the exceptions an
application maps to HTTP responses. The runtime wiring — session management,
persistence, and the pausing tool — is provided by an integration layer such
as ``sefios.fastapi``.
"""

from ._events import (
    SessionEvent,
    SessionEventBroker,
    TokenEventPublisher,
    format_sse_event,
)
from ._human_input import HumanInputCoordinator, HumanInputReceiver, HumanInputStore
from ._kv import KeyValueStore
from ._responses import session_event_response
from .exceptions import (
    AmbiguousHumanInputError,
    InputRequired,
    UnknownHumanInputError,
    UnknownSessionError,
)

__all__ = [
    "KeyValueStore",
    "HumanInputStore",
    "HumanInputReceiver",
    "HumanInputCoordinator",
    "SessionEvent",
    "SessionEventBroker",
    "TokenEventPublisher",
    "format_sse_event",
    "session_event_response",
    "InputRequired",
    "UnknownSessionError",
    "UnknownHumanInputError",
    "AmbiguousHumanInputError",
]
