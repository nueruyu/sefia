"""FastAPI (HTTP) building blocks for Sefia applications.

This package holds HTTP-side pieces that depend only on ``sefia`` and FastAPI.
Per-session SSE streams and application-facing exceptions live in
:mod:`sefia_fastapi.events` and :mod:`sefia_fastapi.exceptions`. Runtime
wiring, input routing, persistence, and pausing tools are provided by an
integration layer such as ``sefios.fastapi``.
"""

__all__: list[str] = []
