"""FastAPI (HTTP) integration for the sefios stack.

The facade over the ``sefia_fastapi`` building blocks: :class:`SefiaHTTP`
wires the HTTP input core to sefios' :class:`Input`, session
storage, cost accounting, and per-session SSE event streams. The surface that
applications need is re-exported here — the ``sefia_fastapi`` HTTP
input-routing errors, plus the core :class:`~sefios.InputRequired` pause the
facade surfaces — so a single ``from sefios.fastapi import ...`` suffices.
"""

from importlib.util import find_spec

try:
    from sefia_fastapi import (
        AmbiguousInputError,
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

from ..exceptions import InputRequired
from ._app import SefiaHTTP, SefiaHTTPSession

__all__ = [
    "SefiaHTTP",
    "SefiaHTTPSession",
    "InputRequired",
    "UnknownSessionError",
    "UnknownInputError",
    "AmbiguousInputError",
]
