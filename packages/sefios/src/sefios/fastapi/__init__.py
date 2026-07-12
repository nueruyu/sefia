"""FastAPI (HTTP) integration for the sefios stack.

The facade over the ``sefia_fastapi`` building blocks: :class:`SefiaHTTP`
wires the HTTP input core to sefios' :class:`InputTool`, session
storage, cost accounting, and per-session SSE token streams. The
``sefia_fastapi`` surface that applications need (exceptions to map to HTTP
responses) is re-exported here, so a single ``from sefios.fastapi import ...``
suffices.
"""

from importlib.util import find_spec

try:
    from sefia_fastapi import (
        AmbiguousInputError,
        InputRequired,
        UnknownInputError,
        UnknownSessionError,
    )
except ImportError as e:
    if find_spec("sefia_fastapi") is None:
        raise ImportError(
            "The 'fastapi' extra is required to use sefios.fastapi. "
            "Please install it with: pip install 'sefios[fastapi]'"
        ) from e
    raise

from ._app import SefiaHTTP, SefiaHTTPSession

__all__ = [
    "SefiaHTTP",
    "SefiaHTTPSession",
    "InputRequired",
    "UnknownSessionError",
    "UnknownInputError",
    "AmbiguousInputError",
]
