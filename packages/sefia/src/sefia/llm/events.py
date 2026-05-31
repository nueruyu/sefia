from __future__ import annotations

from dataclasses import dataclass

from ..events import Event
from .messages import LLMResponse, Message


@dataclass(frozen=True)
class BeforeLLMCall(Event):
    """Event fired just before a call to the LLM."""

    messages: list[Message]
    tools: list[dict] | None
    output_schema: dict | None


@dataclass(frozen=True)
class AfterLLMCall(Event):
    """Event fired just after a response is received from the LLM."""

    response: LLMResponse


@dataclass(frozen=True)
class LLMTokenReceived(Event):
    """Event fired when a token is received from the LLM stream."""

    token: str
