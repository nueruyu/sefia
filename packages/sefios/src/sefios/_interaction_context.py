"""Session-scoped interaction binding."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interactions import InteractionChannel

_active_channel: ContextVar[InteractionChannel] = ContextVar("interaction_channel")


@contextmanager
def bind_interaction_channel(channel: InteractionChannel) -> Generator[None]:
    token = _active_channel.set(channel)
    try:
        yield
    finally:
        _active_channel.reset(token)


def get_interaction_channel() -> InteractionChannel:
    try:
        return _active_channel.get()
    except LookupError:
        raise RuntimeError("Interactions require an active sefios session.") from None
