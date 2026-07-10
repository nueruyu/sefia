"""Typer (CLI) building blocks for Sefia applications.

This package holds the CLI-side pieces that depend only on ``sefia`` and
Typer: the input core (an :class:`InputChannel` persisted over a
:class:`KeyValueStore`), the reporter surface, and the exceptions
applications catch. The runtime wiring — session management, persistence,
and the pausing tool — is provided by an integration layer such as
``sefios.cli``.
"""

from ._input import InputChannel, InputRequest
from ._kv import KeyValueStore
from ._reporter import CLIReporter, DefaultCLIReporter, ResolvedSession
from .exceptions import (
    AmbiguousInputError,
    UnknownInputError,
    UnknownSessionError,
)

__all__ = [
    "InputChannel",
    "InputRequest",
    "KeyValueStore",
    "CLIReporter",
    "DefaultCLIReporter",
    "ResolvedSession",
    "UnknownSessionError",
    "UnknownInputError",
    "AmbiguousInputError",
]
