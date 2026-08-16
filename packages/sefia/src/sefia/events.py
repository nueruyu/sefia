from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .event_system import Event
from .inference import HistoryItem, StepDecision, ToolCallRequest


@dataclass(frozen=True)
class InferenceStart(Event):
    """Event fired when an inference process begins."""

    func_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class BeforeInferenceStep(Event):
    """Event fired just before a call to the inference strategy."""

    history: Sequence[HistoryItem]
    tool_names: list[str]


@dataclass(frozen=True)
class AfterInferenceStep(Event):
    """Event fired just after a decision is received from the inference strategy."""

    decision: StepDecision


@dataclass(frozen=True)
class InferenceStepFailed(Event):
    """
    Event fired when a single inference step raises an exception.

    The exception is re-raised, and glyff leaves the interrupted execution in its
    ``STARTED`` state so it re-runs on resume. This event is for observation
    only: handlers cannot change the outcome, because the publisher isolates
    their exceptions. A resumable interrupt must instead come from the
    control/execution layer (for example, a tool raising
    ``sefia.exceptions.PauseException``).
    """

    error: Exception


@dataclass(frozen=True)
class StepStarted(Event):
    """
    Event fired at the start of every inference step, including the first.

    ``step`` is the 0-based index of the step about to run (equivalently, the
    number of steps already completed in this attempt). This event is for
    observation only; handlers cannot stop the loop, because the publisher
    isolates their exceptions. The loop is bounded by a step middleware (e.g.
    ``StepLimiter``), not by a handler.
    """

    step: int
    history: Sequence[HistoryItem]


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
