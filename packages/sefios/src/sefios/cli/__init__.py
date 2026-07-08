"""Typer (CLI) integration for the sefios stack.

The facade over the ``sefia_typer`` building blocks: :class:`SefiaCLI` wires
the CLI human-input core to sefios' :class:`HumanInputTool`, session storage,
and cost accounting. The ``sefia_typer`` surface that applications need
(reporters, Typer helpers, exceptions) is re-exported here, so a single
``from sefios.cli import ...`` suffices.
"""

try:
    import sefia_typer as _sefia_typer  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The 'cli' extra is required to use sefios.cli. "
        "Please install it with: pip install 'sefios[cli]'"
    ) from e

from sefia_typer import (
    AmbiguousHumanInputError,
    CLIReporter,
    DefaultCLIReporter,
    HumanInputRequest,
    UnknownHumanInputError,
    UnknownSessionError,
    add_session_commands,
    async_command,
)

from ._app import CostReportingCLIReporter, SefiaCLI, SefiaCLISession

__all__ = [
    "SefiaCLI",
    "SefiaCLISession",
    "CLIReporter",
    "DefaultCLIReporter",
    "CostReportingCLIReporter",
    "HumanInputRequest",
    "add_session_commands",
    "async_command",
    "UnknownSessionError",
    "UnknownHumanInputError",
    "AmbiguousHumanInputError",
]
