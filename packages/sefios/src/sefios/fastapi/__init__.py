"""FastAPI (HTTP) integration for the sefios stack.

The facade over the ``sefia_fastapi`` building blocks: :class:`SefiaHTTP`
wires the HTTP input core to sefios' :class:`Input`, session
storage, cost accounting, and per-session SSE event streams. The surface that
applications need is exposed by this subpackage; its exception types live in
:mod:`sefios.fastapi.exceptions`.
"""

from importlib.util import find_spec

try:
    from ._app import SefiaHTTP, SefiaHTTPSession
except ImportError as e:
    if find_spec("sefia_fastapi") is None:
        raise ImportError(
            "The 'fastapi' extra is required to use sefios.fastapi. "
            "Please install it with: pip install 'sefios[fastapi]'"
        ) from e
    raise

__all__ = [
    "SefiaHTTP",
    "SefiaHTTPSession",
]
