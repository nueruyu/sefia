"""Typer (CLI) integration for the sefios stack.

The facade over the ``sefia_typer`` building blocks: :class:`SefiaCLI` wires
the CLI input core to sefios' :class:`InputTool`, session storage,
and cost accounting. The ``sefia_typer`` surface that applications need
(reporters and exceptions) is re-exported here, so a single
``from sefios.cli import ...`` suffices.
"""

from importlib.util import find_spec

try:
    from sefia_typer import (
        AmbiguousInputError,
        CLIReporter,
        DefaultCLIReporter,
        InputRequest,
        ResolvedSession,
        UnknownInputError,
        UnknownSessionError,
    )
except ImportError as e:
    if find_spec("sefia_typer") is None:
        raise ImportError(
            "The 'cli' extra is required to use sefios.cli. "
            "Please install it with: pip install 'sefios[cli]'"
        ) from e
    raise

from ._app import CostReportingCLIReporter, SefiaCLI, SefiaCLISession

__all__ = [
    "SefiaCLI",
    "SefiaCLISession",
    "CLIReporter",
    "DefaultCLIReporter",
    "CostReportingCLIReporter",
    "InputRequest",
    "ResolvedSession",
    "UnknownSessionError",
    "UnknownInputError",
    "AmbiguousInputError",
]
