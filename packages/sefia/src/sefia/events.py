from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import HistoryItem, InferenceDecision, ToolCallRequest


@dataclass(frozen=True)
class Event:
    """Base class for all events."""


@dataclass(frozen=True)
class InferenceStart(Event):
    """Event fired when an inference process begins."""

    func_name: str
    args: tuple
    kwargs: dict


@dataclass(frozen=True)
class BeforeInferenceStep(Event):
    """Event fired just before a call to the inference strategy."""

    history: list[HistoryItem]
    tools: list[dict]


@dataclass(frozen=True)
class AfterInferenceStep(Event):
    """Event fired just after a decision is received from the inference strategy."""

    decision: InferenceDecision


@dataclass(frozen=True)
class NextTurnRequested(Event):
    """
    Event fired before the inference loop would take another turn.

    The executor does not loop on its own: after a step that is not a final
    answer, it asks whether another turn is permitted. A handler grants the
    next turn by raising ``RequestNextTurn``; if no handler does, the loop
    ends with ``MaxTurnsExceededError``. ``completed_turns`` is the number of
    inference steps already taken in the current attempt.
    """

    completed_turns: int
    history: list[HistoryItem]


@dataclass(frozen=True)
class BeforeToolCall(Event):
    """Event fired just before a tool is executed."""

    tool_call: ToolCallRequest


@dataclass(frozen=True)
class AfterToolCall(Event):
    """Event fired after a tool executes successfully."""

    tool_call: ToolCallRequest
    result: Any


@dataclass(frozen=True)
class ToolExecutionFailed(Event):
    """Event fired when a tool execution fails with an exception."""

    tool_call: ToolCallRequest
    error: Exception


@dataclass(frozen=True)
class InferenceEnd(Event):
    """Event fired when an inference process completes."""

    result: Any


@dataclass(frozen=True)
class InferenceFailed(Event):
    """Event fired when an inference attempt fails with an exception (e.g. validation error)."""

    error: Exception


@dataclass(frozen=True)
class AttemptStart(Event):
    """Event fired at the beginning of each inference attempt."""
