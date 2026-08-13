"""Exceptions surfaced by the sefios CLI integration."""

from sefia_typer.exceptions import (
    AmbiguousInputError,
    UnknownInputError,
    UnknownSessionError,
)

__all__ = [
    "UnknownSessionError",
    "UnknownInputError",
    "AmbiguousInputError",
]
