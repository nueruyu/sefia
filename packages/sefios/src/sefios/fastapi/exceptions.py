"""Exceptions surfaced by the sefios FastAPI integration."""

from sefia_fastapi.exceptions import (
    AmbiguousInputError,
    UnknownInputError,
    UnknownSessionError,
)

from ..exceptions import InputRequired

__all__ = [
    "InputRequired",
    "UnknownSessionError",
    "UnknownInputError",
    "AmbiguousInputError",
]
