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
class InferenceStepFailed(Event):
    """
    Event fired when a single inference step raises an exception.

    The step is engraved, so by default the exception is re-raised and
    persisted by glyff as a permanent failure. A handler may instead raise
    ``glyff.exceptions.YieldException`` to interrupt the session gracefully and
    leave the step resumable (nothing is persisted). sefia does not decide
    whether the error is recoverable; that policy lives in the handler.
    """

    error: Exception


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
