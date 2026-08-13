"""Exceptions surfaced by the sefios CLI integration."""

from sefia_typer.exceptions import UnknownSessionError

from ..exceptions import AmbiguousInputError, UnknownInputError

__all__ = [
    "UnknownSessionError",
    "UnknownInputError",
    "AmbiguousInputError",
]
