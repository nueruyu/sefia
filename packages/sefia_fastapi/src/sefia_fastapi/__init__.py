"""FastAPI (HTTP) building blocks for Sefia applications.

This package holds the HTTP-side pieces that depend only on ``sefia`` and
FastAPI: the input core (an :class:`InputChannel` persisted over a
:class:`KeyValueStore`), per-session SSE streams (:class:`SessionEvents`),
and the exceptions an application maps to HTTP responses. The runtime
wiring — session management, persistence, and the pausing tool — is provided
by an integration layer such as ``sefios.fastapi``.
"""

from ._events import SessionEvents, SSEEvent
from ._input import InputChannel, InputRequest
from ._kv import KeyValueStore
from .exceptions import (
    AmbiguousInputError,
    InputRequired,
    UnknownInputError,
    UnknownSessionError,
)

__all__ = [
    "InputChannel",
    "InputRequest",
    "KeyValueStore",
    "SessionEvents",
    "SSEEvent",
    "InputRequired",
    "UnknownSessionError",
    "UnknownInputError",
    "AmbiguousInputError",
]
