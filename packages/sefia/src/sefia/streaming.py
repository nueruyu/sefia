"""Streaming of a tool call's arguments as they are decoded from the LLM.

A tool may opt in to receive its arguments incrementally, as the model emits
them, via a side channel registered with :func:`sefia.preview`. The handler
is given an :data:`ArgStream` — an async iterator of :data:`ArgEvent` — and can
react (for example, render a question to a UI) before the full tool call has
been decoded.

This is a best-effort *preview* channel layered over the durable inference
flow. The authoritative arguments are still the fully decoded values the tool
itself receives; the stream never becomes the argument. On replay the decode
step is served from glyff's cache and the side channel simply does not re-fire,
so live output is never replayed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class StringDelta:
    """An incremental chunk of a string argument's value."""

    name: str
    text: str


@dataclass(frozen=True)
class StringEnd:
    """A string argument completed, carrying its fully assembled value."""

    name: str
    value: str


@dataclass(frozen=True)
class Scalar:
    """A non-string scalar argument, delivered whole."""

    name: str
    value: int | float | bool | None


ArgEvent: TypeAlias = StringDelta | StringEnd | Scalar
ArgStream: TypeAlias = AsyncIterator[ArgEvent]
StreamHandler: TypeAlias = Callable[[str, ArgStream], Awaitable[None]]
