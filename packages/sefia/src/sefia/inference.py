from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRequest:
    """Represents a request to call a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResult:
    """Represents the result of a single tool call."""

    tool_call_id: str
    result: Any


@dataclass
class ToolCallDecision:
    """A decision to call one or more tools."""

    calls: list[ToolCallRequest]


@dataclass
class FinalAnswerDecision:
    """A decision to return the final answer."""

    answer: Any


InferenceDecision = ToolCallDecision | FinalAnswerDecision
HistoryItem = ToolCallDecision | ToolCallResult


@dataclass
class InferenceHistory:
    """Represents the persisted history of an inference process."""

    items: list[HistoryItem] = field(default_factory=list)


__all__ = [
    "ToolCallRequest",
    "ToolCallResult",
    "ToolCallDecision",
    "FinalAnswerDecision",
    "InferenceDecision",
    "HistoryItem",
    "InferenceHistory",
]
