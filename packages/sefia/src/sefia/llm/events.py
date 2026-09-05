from __future__ import annotations

from dataclasses import dataclass

from ..event_system import Event
from ..exceptions import InvalidInferenceResponseError
from ._messages import LLMCompletion
from .step_decision import DecisionSpec


@dataclass(frozen=True)
class BeforeLLMCall(Event):
    """Event fired just before a call to the LLM."""

    prompt: str
    decision_spec: DecisionSpec


@dataclass(frozen=True)
class DecisionRepairAttempt(Event):
    """
    Event fired when an invalid LLM decision is about to be retried with
    corrective feedback included in a newly rendered prompt.

    ``attempt`` is the 1-based number of the repair attempt about to run.
    """

    error: InvalidInferenceResponseError
    attempt: int


@dataclass(frozen=True)
class AfterLLMCall(Event):
    """Event fired just after a completion is received from the LLM."""

    completion: LLMCompletion


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
