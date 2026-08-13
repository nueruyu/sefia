"""Typer (CLI) building blocks for Sefia applications.

This package holds CLI reporting types and exceptions that depend only on
``sefia`` and Typer. Runtime wiring, input routing, persistence, and pausing
tools are provided by an integration layer such as ``sefios.cli``.
"""

from ._reporter import (
    CLIReporter,
    DefaultCLIReporter,
    InputRequest,
    OutputMessage,
    ResolvedSession,
)

__all__ = [
    "InputRequest",
    "CLIReporter",
    "DefaultCLIReporter",
    "OutputMessage",
    "ResolvedSession",
]
