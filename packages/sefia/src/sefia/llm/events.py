from __future__ import annotations

from dataclasses import dataclass

from ..event_system import Event
from ._messages import LLMResponse, Message


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
    """Event fired when a content token is received from the LLM stream."""

    token: str


@dataclass(frozen=True)
class LLMReasoningTokenReceived(Event):
    """Event fired when a reasoning (thinking) token is received from the LLM stream.

    Reasoning is step-scoped: it is the model's thinking for the current
    inference step, emitted before the step's response content, and is not tied
    to any particular tool call.
    """

    token: str
