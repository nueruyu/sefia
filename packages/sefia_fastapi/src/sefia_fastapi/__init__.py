"""FastAPI (HTTP) building blocks for Sefia applications.

This package holds the HTTP-side pieces that depend only on ``sefia`` and
FastAPI: the input core (an :class:`InputChannel` persisted over a
:class:`KeyValueStore`). Per-session SSE streams and application-facing
exceptions live in :mod:`sefia_fastapi.events` and
:mod:`sefia_fastapi.exceptions`. The runtime
wiring — session management, persistence, and the pausing tool — is provided
by an integration layer such as ``sefios.fastapi``.
"""

from sefia.input_channels import InputChannel, InputRequest, KeyValueStore

__all__ = [
    "InputChannel",
    "InputRequest",
    "KeyValueStore",
]
