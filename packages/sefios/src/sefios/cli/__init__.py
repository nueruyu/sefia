"""Typer (CLI) integration for the sefios stack.

The facade over the ``sefia_typer`` building blocks: :class:`SefiaCLI` wires
the CLI input core to sefios' :class:`Input`, session storage,
and cost accounting. Reporter types are re-exported here; exceptions live in
:mod:`sefios.cli.exceptions`.
"""

from importlib.util import find_spec

try:
    from sefia_typer import (
        CLIReporter,
        DefaultCLIReporter,
        InputRequest,
        ResolvedSession,
    )
except ImportError as e:
    if find_spec("sefia_typer") is None:
        raise ImportError(
            "The 'cli' extra is required to use sefios.cli. "
            "Please install it with: pip install 'sefios[cli]'"
        ) from e
    raise

from ._app import SefiaCLI, SefiaCLISession
from ._cost_reporter import CostReportingCLIReporter

__all__ = [
    "SefiaCLI",
    "SefiaCLISession",
    "CLIReporter",
    "DefaultCLIReporter",
    "CostReportingCLIReporter",
    "InputRequest",
    "ResolvedSession",
]
