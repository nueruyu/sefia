"""Typer (CLI) building blocks for Sefia applications.

This package holds the CLI-side pieces that depend only on ``sefia`` and
Typer: the human-in-the-loop core (persisted over a :class:`KeyValueStore`),
the reporter surface, and Typer helpers. The runtime wiring — session
management, persistence, and the pausing tool — is provided by an integration
layer such as ``sefios.cli``.
"""

from ._human_input import (
    HumanInputCoordinator,
    HumanInputReceiver,
    HumanInputRequest,
    HumanInputStore,
)
from ._kv import KeyValueStore
from ._reporter import CLIReporter, DefaultCLIReporter, ResolvedSessionLike
from ._typer_utils import SessionCommands, add_session_commands, async_command
from .exceptions import (
    AmbiguousHumanInputError,
    UnknownHumanInputError,
    UnknownSessionError,
)

__all__ = [
    "KeyValueStore",
    "HumanInputStore",
    "HumanInputReceiver",
    "HumanInputCoordinator",
    "HumanInputRequest",
    "CLIReporter",
    "DefaultCLIReporter",
    "ResolvedSessionLike",
    "SessionCommands",
    "add_session_commands",
    "async_command",
    "UnknownSessionError",
    "UnknownHumanInputError",
    "AmbiguousHumanInputError",
]
