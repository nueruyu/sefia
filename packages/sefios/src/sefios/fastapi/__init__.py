"""FastAPI (HTTP) integration for the sefios stack.

The facade over the ``sefia_fastapi`` building blocks: :class:`SefiaHTTP`
wires the HTTP human-input core to sefios' :class:`HumanInputTool`, session
storage, cost accounting, and per-session SSE token streams. The
``sefia_fastapi`` surface that applications need (exceptions to map to HTTP
responses) is re-exported here, so a single ``from sefios.fastapi import ...``
suffices.
"""

try:
    import sefia_fastapi as _sefia_fastapi  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The 'fastapi' extra is required to use sefios.fastapi. "
        "Please install it with: pip install 'sefios[fastapi]'"
    ) from e

from sefia_fastapi import (
    AmbiguousHumanInputError,
    InputRequired,
    UnknownHumanInputError,
    UnknownSessionError,
)

from ._app import SefiaHTTP, SefiaHTTPSession

__all__ = [
    "SefiaHTTP",
    "SefiaHTTPSession",
    "InputRequired",
    "UnknownSessionError",
    "UnknownHumanInputError",
    "AmbiguousHumanInputError",
]
