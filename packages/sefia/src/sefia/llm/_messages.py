from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .structured_data import StructuredData


@dataclass
class Message:
    """Represents a single message in a conversation with an LLM."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Any] | None = None
    tool_call_id: str | None = None  # Required for role="tool" messages
    tool_calls: list[ToolCall] | None = None


@dataclass
class ToolCall:
    """Represents a tool call requested by the LLM."""

    id: str
    name: str
    arguments: StructuredData


@dataclass
class LLMCompletion:
    """A provider-neutral completion returned by an ``LLMClient``."""

    model: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None
    cost: float | None = None
    structured_output: StructuredData | None = None
